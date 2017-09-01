
// System includes
#include <stdio.h>
#include <assert.h>
#include <iostream>

// CUDA runtime
#include <cuda_runtime.h>

#include <fstream>
#include <ctime>
#include <chrono>
#include <thread>

#include "AIAConstants.h"
#include "firdem.h"
#include "../../utils/intTabulated.h"
#include "../../utils/cuda_inverse.h"

#include <log4cxx/logger.h>



static log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("scidb.firdem"));


// Helper Macros for cleaner GPU code
#define __tx					  		(threadIdx.x)
#define __ty					  		(threadIdx.y)
#define __PRINT_D(name,arg)       		printf("%s B%d y%d x%d : %.17g\n",#name,blockIdx.x,threadIdx.y,threadIdx.x,arg);
#define __PRINT_I(name,arg)       		printf("%s B%d y%d x%d : %d\n",#name,blockIdx.x,threadIdx.y,threadIdx.x,arg);
#define MAX(a, b) 				  		(((a) > (b)) ? (a) : (b))

#define __SIZE_D_CONST_NCHAN			(3*t_nchan									)
#define __D_CONST_DNPP(arg)				(d_const_nchan[ (0*nchan) + arg			])
#define __D_CONST_DNPP_ADDR 		    (&d_const_nchan[(0*t_nchan)    				])
#define __D_CONST_RDN(arg)				(d_const_nchan[ (1*nchan) + arg			])
#define __D_CONST_RDN_ADDR 		    	(&d_const_nchan[(1*t_nchan)					])
#define __D_CONST_TR_NORM(arg)			(d_const_nchan[ (2*nchan) + arg			])
#define __D_CONST_TR_NORM_ADDR	    	(&d_const_nchan[(2*t_nchan)					])

#define SIZE_D_NCHAN_INT_BLOCK			(2*t_nchan*chunk_size							)
#define D_DATAIN_BLOCK(arg)				(d_nchan_int[(0*chunk_size*nchan)+(blockIdx.x*nchan)+arg])
#define D_ERRSIN_BLOCK(arg)				(d_nchan_int[(1*chunk_size*nchan)+(blockIdx.x*nchan)+arg])
#define D_DATAIN_ADDR_BLOCK(stream)		(&((d_nchan_int[stream])[(0*chunk_size*t_nchan)]))
#define D_ERRSIN_ADDR_BLOCK(stream)		(&((d_nchan_int[stream])[(1*chunk_size*t_nchan)]))




#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)












// calculate the emission weighted median temperature (EMWMT)
//  and total emission measure (EMTOT)
template <typename t_precision, typename T, int nchan, int nb2> __global__ void
__launch_bounds__(192, 9) firdem_emwtemp(
	t_precision* d_BASIS,
	t_precision* d_logt,
	t_precision* d_dem_coofs
	){

	__shared__ t_precision s_helper[331];
	__shared__ t_precision s_helper2[331];
	double r_helper = 0.0;

    if (threadIdx.x <nb2 && threadIdx.y == 0){
    	s_helper2[__tx]=d_dem_coofs[blockIdx.x*(nb2+3)+threadIdx.x];
    }
    __syncthreads();

	#pragma unroll
    for(int i = 0; i < 55; ++i){
    	r_helper=d_BASIS[(i*nchan*nb2)+(threadIdx.y*nb2)+__tx]*s_helper2[__tx];
		#pragma unroll
		for (int mask = 16; mask > 0; mask /= 2){
			r_helper += __shfl_down(r_helper, mask);
		}
		if(__tx==0){
			s_helper[(i*nchan)+threadIdx.y]=r_helper;
		}
    }
    if(__ty==0){
    	r_helper=d_BASIS[(330*nb2)+__tx]*s_helper2[__tx];
		#pragma unroll
		for (int mask = 16; mask > 0; mask /= 2){
			r_helper += __shfl_down(r_helper, mask);
		}
		if(__tx==0){
			s_helper[330]=r_helper;
		}
    }


    __syncthreads();
    // Computes the median of a tabulated distribution function
    s_helper2[1+(__ty*nb2)+__tx] = d_logt[(__ty*nb2)+__tx+1]-d_logt[(__ty*nb2)+__tx];

    if(__ty*nb2+__tx+(nb2*nchan) < 330){
    	s_helper2[1+(nb2*nchan)+(__ty*nb2)+__tx] = d_logt[(nb2*nchan)+(__ty*nb2)+__tx+1]-d_logt[(nb2*nchan)+(__ty*nb2)+__tx];
    }
    //printf("yavg B%d i%d  %.17g \n",blockIdx.x,1+(__ty*nb2)+__tx,
    //		(0.5*(d_basis_times_coffs[(blockIdx.x*331)+(__ty*nb2)+__tx+1]+d_basis_times_coffs[(blockIdx.x*331)+(__ty*nb2)+__tx])) );
    s_helper2[1+(__ty*nb2)+__tx] = s_helper2[1+(__ty*nb2)+__tx] * (0.5*(s_helper[(__ty*nb2)+__tx+1]+s_helper[(__ty*nb2)+__tx]));
    //printf("sum B%d i%d  %.17g \n",blockIdx.x,1+(__ty*nb2)+__tx,
    //		s_helper2[1+(__ty*nb2)+__tx]);
    if(__ty*nb2+__tx+(nb2*nchan) < 330){
    	s_helper2[1+(nb2*nchan)+(__ty*nb2)+__tx] = s_helper2[1+(nb2*nchan)+(__ty*nb2)+__tx] * (
    			0.5*(s_helper[(nb2*nchan)+(__ty*nb2)+__tx+1]+s_helper[(nb2*nchan)+(__ty*nb2)+__tx]));
        //printf("sum B%d i%d  %.17g \n",blockIdx.x,1+(nb2*nchan)+(__ty*nb2)+__tx,
        //		s_helper2[1+(nb2*nchan)+(__ty*nb2)+__tx] );
    }

    __syncthreads();
    //create the cumulative sum
    if(__tx == 0 and __ty == 0){
    	s_helper2[0]=0;
		#pragma unroll
		for(int i = 2;i<331;i++){
			s_helper2[i]+=s_helper2[i-1];
		}
		// in case we have all zeros, return
		if (s_helper2[330]==0.0){
		    d_dem_coofs[blockIdx.x*(nb2+3)+nb2+1]=0.0;
		    d_dem_coofs[blockIdx.x*(nb2+3)+nb2+2]=0.0;
		}else{
			d_dem_coofs[blockIdx.x*(nb2+3)+nb2+1] = s_helper2[330];
		}
    }
    __syncthreads();
	if (s_helper2[330]==0.0){return;}

	s_helper2[(__ty*nb2)+__tx]=s_helper2[(__ty*nb2)+__tx]/s_helper2[330];
	if(__ty*nb2+__tx+(nb2*nchan) < 331){
		s_helper2[(nb2*nchan)+(__ty*nb2)+__tx] =s_helper2[(nb2*nchan)+(__ty*nb2)+__tx]/s_helper2[330];
	}
	__syncthreads();

	//find closest integral and interpolate
	if(s_helper2[(__ty*nb2)+__tx] <= 0.5 and s_helper2[(__ty*nb2)+__tx+1] > 0.5 ){
		s_helper[0]=d_logt[(__ty*nb2)+__tx+1]-d_logt[(__ty*nb2)+__tx];
		s_helper[1]=(0.5-s_helper2[(__ty*nb2)+__tx]) *
			s_helper[0]/(s_helper2[(__ty*nb2)+__tx+1]-
			s_helper2[(__ty*nb2)+__tx]) + d_logt[(__ty*nb2)+__tx];

		d_dem_coofs[blockIdx.x*(nb2+3)+nb2+2] =s_helper[1];
/*		printf("firdem_emwtemp_image B%d y%d x%d  %.17g %.17g \n",blockIdx.x,threadIdx.y, threadIdx.x,
				d_dem_coofs[blockIdx.x*(nb2+3)+nb2+1], s_helper[1]);*/
	}
	if(__ty*nb2+__tx+(nb2*nchan) < 330){
		if(s_helper2[(nb2*nchan)+(__ty*nb2)+__tx] <= 0.5 and s_helper2[(nb2*nchan)+(__ty*nb2)+__tx] > 0.5 ){
			s_helper[0]=d_logt[(nb2*nchan)+(__ty*nb2)+__tx+1]-d_logt[(nb2*nchan)+(__ty*nb2)+__tx];
			s_helper[1]=(0.5-s_helper2[(nb2*nchan)+(__ty*nb2)+__tx]) *
				s_helper[0]/(s_helper2[(nb2*nchan)+(__ty*nb2)+__tx+1]-
				s_helper2[(nb2*nchan)+(__ty*nb2)+__tx]) + d_logt[(nb2*nchan)+(__ty*nb2)+__tx];

			d_dem_coofs[blockIdx.x*(nb2+3)+nb2+2] =s_helper[1];
		}
	}
}




template <typename t_precision, typename T, int nchan, int nb2> __device__ void firdem_iterate(
		t_precision s_broadcast_vars[],
		//[0]=wsmall / break_loop / chi2_iter_thold
		//[1]=nb_bad
		//[2]=chi2min
		//[3]=chi2_noextrap
		//[4]=itcount
		//[5]=lastchi2
		//[6-7] used by outer loop
		//[8]=chi2
		//[9]=chi2_test
		t_precision s_helper_nb2[],
		t_precision s_helper_nchan[],
		t_precision s_helper_nchan2[],
		t_precision s_helper_nchan3[],
		t_precision s_helper_nchan_nchan[],
		t_precision s_helper_nchan_nchan2[],
		t_precision s_helper_nchan_nb2[],
		int k,
		t_precision* d_minconds,
		t_precision* d_WVEC,
		t_precision* d_tr_norm2,
		t_precision* d_dempos,
		t_precision* d_dempos_out,
		t_precision* d_ainv_arr,
		t_precision* d_datapos,
		t_precision* d_a2_array,
		t_precision* d_deltadata,
		int16_t* d_nchan_int,
		t_precision* d_deltatot,
		int32_t chunk_size,
		t_precision* d_basis22,
		t_precision* d_dem,
		t_precision* d_data,
		t_precision* d_dem_noextrap,
		t_precision* d_lastdem,
		t_precision* d_dem_out,
		t_precision* d_dem_test,
		t_precision* d_dempos_test,
		t_precision* d_last_dem,
		t_precision* d_a_inv_arr,
		t_precision* d_chi2thold_iter
){

	t_precision r_helper = 0.0;


	if (threadIdx.x == 0 && threadIdx.y == 0){
		s_broadcast_vars[1] = 0.0;
		s_broadcast_vars[4] = 0.0;
		s_broadcast_vars[5] = -1.0;
	}
	if(threadIdx.x < nchan && threadIdx.y < nchan){
		s_helper_nchan_nchan[threadIdx.y*nchan + threadIdx.x]=
				d_a_inv_arr[(k*nchan*nchan)+(threadIdx.y*nchan)+threadIdx.x ];
	}
	__syncthreads();

	//dempos = dem > 0 | dem is still s_helper_nb2
	if(threadIdx.x < nb2 && threadIdx.y == 0){
		if(s_helper_nb2[threadIdx.x]<0.0){s_helper_nb2[threadIdx.x]= 0.0;}
		d_dempos[(nb2*blockIdx.x)+threadIdx.x]=s_helper_nb2[threadIdx.x];
		d_dempos_out[(nb2*blockIdx.x)+threadIdx.x]=s_helper_nb2[threadIdx.x];
		d_lastdem[(nb2*blockIdx.x) + threadIdx.x]=0.0;
	}
	__syncthreads();


	//Data values for DEM with negatives zeroed
	//datapos = reform(a2_array#dempos)
	r_helper=d_a2_array[threadIdx.x + threadIdx.y*nb2]*s_helper_nb2[threadIdx.x];
	__syncthreads();
	#pragma unroll
	for (int mask = 16; mask > 0; mask /= 2){
		r_helper += __shfl_down(r_helper, mask);
	}
	if(__tx==0){s_helper_nchan[threadIdx.y]=r_helper;}
	__syncthreads();


	//deltadata = datapos-data | data is still stored in s_helper_nchan3
	if (threadIdx.x < nchan && threadIdx.y == 0){
		s_helper_nchan3[threadIdx.x]= s_helper_nchan[threadIdx.x]-s_helper_nchan3[threadIdx.x];
		d_deltadata[(nchan*blockIdx.x)+threadIdx.x]=s_helper_nchan3[threadIdx.x];
	}
	__syncthreads();

	//deltatot(0,*) = ((datapos)/sigmdemas)^2
	if(threadIdx.x < nchan && threadIdx.y == 0){
		s_helper_nchan[threadIdx.x]=s_helper_nchan[threadIdx.x]/
				D_ERRSIN_BLOCK(__tx);
		s_helper_nchan[threadIdx.x]=s_helper_nchan[threadIdx.x]*s_helper_nchan[threadIdx.x];
		d_deltatot[(nchan*blockIdx.x)+threadIdx.x]=s_helper_nchan[threadIdx.x];
	}
	__syncthreads();

	//chi2min = deltatot^2#chi2totarr
	if(threadIdx.x == 0 && threadIdx.y == 0){
		s_broadcast_vars[2]=0.0;
		#pragma unroll
		for (int i = 0; i < nchan; ++i){
			s_broadcast_vars[2]+=s_helper_nchan[i]*s_helper_nchan[i]*(1.0/nchan);
		}
	}
	__syncthreads();


	for (int itcount = 0; itcount < 1000; itcount++){ //1000

		//Stop if we've gone over nbad_recent points since last chi squared improvement
		__syncthreads();
		if(s_broadcast_vars[1]> 250){break;}

		// Compute DEM correction coefficients in the instrument response basis.
		// These will restore the DEM to agreement with the data, but generally
		// introduce some negative emission.
		if (threadIdx.x < nchan && threadIdx.y < nchan){
			s_helper_nchan_nchan2[threadIdx.y*nchan + threadIdx.x]=
					s_helper_nchan_nchan[threadIdx.y*nchan + threadIdx.x]*
					(d_deltadata[(nchan*blockIdx.x)+threadIdx.x]/
					d_tr_norm2[(nchan*blockIdx.x)+threadIdx.x]);
		}
		__syncthreads();
		if (threadIdx.x < nchan && threadIdx.y == 0){
			r_helper = 0.0;
			#pragma unroll
    		for (int i = 0; i < nchan; ++i){
    			r_helper += s_helper_nchan_nchan2[threadIdx.x*nchan + i];
    		}
    		s_helper_nchan3[threadIdx.x] = r_helper;
		}
		__syncthreads();


		s_helper_nchan_nb2[threadIdx.y*nb2 + threadIdx.x]=d_basis22[threadIdx.y*nb2 + threadIdx.x]*s_helper_nchan3[(threadIdx.y*nb2+threadIdx.x)%nchan];
		__syncthreads();
		if (threadIdx.x < nb2 && threadIdx.y == 0){
			s_helper_nb2[threadIdx.x]=0.0;
			#pragma unroll
			for (int i = 0; i < nchan; ++i){
				s_helper_nb2[threadIdx.x]+=s_helper_nchan_nb2[threadIdx.x*nchan + i];
			}
		}
		__syncthreads();


		//dem(*) = dempos - deltadem
		if(threadIdx.x < nb2 && threadIdx.y == 0){
			s_helper_nb2[threadIdx.x]=d_dempos[(nb2*blockIdx.x)+threadIdx.x]-s_helper_nb2[threadIdx.x];
			d_dem[blockIdx.x*nb2 + threadIdx.x]=s_helper_nb2[threadIdx.x];
			//dem_noextrap=dem
			d_dem_noextrap[blockIdx.x*nb2 + threadIdx.x]=s_helper_nb2[threadIdx.x];
			if(s_helper_nb2[threadIdx.x]<0.0){s_helper_nb2[threadIdx.x]= 0.0;}
			d_dempos[(nb2*blockIdx.x)+threadIdx.x]=s_helper_nb2[threadIdx.x];
		}
		__syncthreads();


		r_helper=d_a2_array[threadIdx.x + threadIdx.y*nb2]*s_helper_nb2[threadIdx.x];
		__syncthreads();
		#pragma unroll
		for (int mask = 16; mask > 0; mask /= 2){
			r_helper += __shfl_down(r_helper, mask);
		}
		if(__tx==0){s_helper_nchan[threadIdx.y]=r_helper;}
		__syncthreads();

		//deltadata = datapos-data
		if (threadIdx.x < nchan && threadIdx.y == 0){
			s_helper_nchan3[threadIdx.x]= s_helper_nchan[threadIdx.x]-d_data[(blockIdx.x*nchan)+threadIdx.x];
			d_deltadata[(nchan*blockIdx.x)+threadIdx.x]=s_helper_nchan3[threadIdx.x];

		}
		__syncthreads();

		//deltatot(0,*) = ((datapos)/sigmdemas)^2
		if(threadIdx.x < nchan && threadIdx.y == 0){
			s_helper_nchan[threadIdx.x]=s_helper_nchan3[threadIdx.x]/
					D_ERRSIN_BLOCK(__tx);
			s_helper_nchan[threadIdx.x]=s_helper_nchan[threadIdx.x]*s_helper_nchan[threadIdx.x];
			d_deltatot[(nchan*blockIdx.x)+threadIdx.x]=s_helper_nchan[threadIdx.x];
		}
		__syncthreads();

		//chi2 = deltatot#chi2totarr
		if(threadIdx.x == 0 && threadIdx.y == 0){
			s_broadcast_vars[8]=0.0;
			#pragma unroll
			for (int i = 0; i < nchan; ++i){
				s_broadcast_vars[8]+=s_helper_nchan[i]*(1.0/nchan);
			}
			s_broadcast_vars[3]=s_broadcast_vars[8];

		}
		__syncthreads();


		if (itcount >= 50+1){
			//If current chi squared is better than the previous one, extrapolate.
			//Otherwise, do nothing
			if (s_broadcast_vars[8] < s_broadcast_vars[5]){

				//load data to shared memory
				if (threadIdx.x < nb2 && threadIdx.y==0){
					s_helper_nchan_nb2[threadIdx.x]=d_dem[blockIdx.x*nb2 + threadIdx.x];
				}
				if (threadIdx.x < nb2 && threadIdx.y==1){
					s_helper_nchan_nb2[threadIdx.x+nb2]=d_lastdem[(nb2*blockIdx.x) + threadIdx.x];
				}
				__syncthreads();

				//Simple linear extrapolation scheme: Change is computed between
				//the most recent DEM and the previous DEM (before extrapolation).
				//This is multiplied by extrap_fac*itcount and added to the most
				//recent DEM
				if (threadIdx.x < nb2 && threadIdx.y==0){
					s_helper_nb2[threadIdx.x] = (s_helper_nchan_nb2[threadIdx.x]-
							s_helper_nchan_nb2[threadIdx.x+nb2])*0.02*itcount;

					s_helper_nb2[threadIdx.x] = s_helper_nchan_nb2[threadIdx.x]+s_helper_nb2[threadIdx.x];
					d_dem_test[blockIdx.x*nb2 + threadIdx.x]=s_helper_nb2[threadIdx.x];

					if(s_helper_nb2[threadIdx.x]<0.0){s_helper_nb2[threadIdx.x]=0.0;}
					d_dempos_test[blockIdx.x*nb2 + threadIdx.x]=s_helper_nb2[threadIdx.x];

				}
				__syncthreads();


				//Compute difference between data for positive DEM and initial data
				//datapos(*) = a2_array#dempos
				r_helper=d_a2_array[threadIdx.x + threadIdx.y*nb2]*s_helper_nb2[threadIdx.x];
				__syncthreads();
				#pragma unroll
				for (int mask = 16; mask > 0; mask /= 2){
					r_helper += __shfl_down(r_helper, mask);
				}
				if(__tx==0){s_helper_nchan[threadIdx.y]=r_helper;}
				__syncthreads();

				//deltadata = datapos-data
				if (threadIdx.x < nchan && threadIdx.y == 0){
					s_helper_nchan3[threadIdx.x]= s_helper_nchan[threadIdx.x]-d_data[(blockIdx.x*nchan)+threadIdx.x];
				}
				__syncthreads();

				//deltatot(0,*) = ((datapos)/sigmdemas)^2
				if(threadIdx.x < nchan && threadIdx.y == 0){
					s_helper_nchan2[threadIdx.x]=s_helper_nchan3[threadIdx.x]/
							D_ERRSIN_BLOCK(__tx);
					s_helper_nchan2[threadIdx.x]=s_helper_nchan2[threadIdx.x]*s_helper_nchan2[threadIdx.x];
					d_deltatot[(nchan*blockIdx.x)+threadIdx.x]=s_helper_nchan2[threadIdx.x];
				}
				__syncthreads();

				//chi2 = deltatot#chi2totarr
				if(threadIdx.x == 0 && threadIdx.y == 0){
					s_broadcast_vars[9]=0.0;
					#pragma unroll
					for (int i = 0; i < nchan; ++i){
						s_broadcast_vars[9]+=s_helper_nchan2[i]*(1.0/nchan);
					}
				}
				__syncthreads();

				//If the resulting chi squared is better than the previous one,
				//update the DEM with the extrapolation
				if(threadIdx.x < nb2 && threadIdx.y == 0){
					if(s_broadcast_vars[9]<s_broadcast_vars[8]){
						if(threadIdx.x == 0 && threadIdx.y == 0){
							s_broadcast_vars[8]=s_broadcast_vars[9];
						}
						if(threadIdx.x < nchan && threadIdx.y == 0){
							d_datapos[(nchan*blockIdx.x)+threadIdx.x]=s_helper_nchan[threadIdx.x];
							d_deltadata[(nchan*blockIdx.x)+threadIdx.x]=s_helper_nchan3[threadIdx.x];
						}
						if(threadIdx.x < nb2 && threadIdx.y == 0){
							d_dem[(nb2*blockIdx.x)+threadIdx.x]=d_dem_test[(nb2*blockIdx.x)+threadIdx.x];
							d_dempos[blockIdx.x*nb2 + threadIdx.x]=s_helper_nb2[threadIdx.x];
						}
					}
				}
				__syncthreads();

			}
		}
		__syncthreads();

		//If chi squared got better, update output chi squared and dems
		if (s_broadcast_vars[8] < s_broadcast_vars[2]){

			__syncthreads();

			if(threadIdx.x == 0 && threadIdx.y == 0){
				s_broadcast_vars[2] = s_broadcast_vars[8];
				s_broadcast_vars[1] = 0.0;
			}
			if(threadIdx.x <nb2 && threadIdx.y == 1){
				d_dempos_out[blockIdx.x*nb2 + threadIdx.x] = d_dempos[blockIdx.x*nb2 + threadIdx.x];
				d_dem_out[blockIdx.x*nb2 + threadIdx.x] = d_dem[blockIdx.x*nb2 + threadIdx.x];
			}
			__syncthreads();
			if(s_broadcast_vars[8] < 1.7312430540720622){break;}

		}
		__syncthreads();

		if(threadIdx.x <nb2 && threadIdx.y == 1){
			d_lastdem[blockIdx.x*nb2 + threadIdx.x] = d_dem_noextrap[blockIdx.x*nb2 + threadIdx.x];
		}
		if(threadIdx.x ==0 && threadIdx.y == 0){
			s_broadcast_vars[1]+=1.0;
			s_broadcast_vars[5]=s_broadcast_vars[3];
		}
		__syncthreads();

	}//endfor


	//Finished iterating. dem contains the DEM corresponding to the best chi squared
	//(with negatives zeroed) found. Although chi squared is calculated with negatives
	//zeroed, the returned DEM retains them for good measure (the user will generally
	//want to zero the neitcountgatives before using the DEMs; a trivial exercise)
	__syncthreads();
	r_helper=d_a2_array[threadIdx.x + threadIdx.y*nb2]*d_dempos_out[blockIdx.x*nb2 + threadIdx.x];
	__syncthreads();
	#pragma unroll
	for (int mask = 16; mask > 0; mask /= 2){
		r_helper += __shfl_down(r_helper, mask);
	}
	if(__tx==0){s_helper_nchan[threadIdx.y]=r_helper;}
	__syncthreads();


	//deltadata = datapos-data
	if (threadIdx.x < nchan && threadIdx.y == 0){
		s_helper_nchan3[threadIdx.x]= s_helper_nchan[threadIdx.x]-d_data[(blockIdx.x*nchan)+threadIdx.x];
	}
	__syncthreads();

	//deltatot(0,*) = ((datapos)/sigmdemas)^2
	if(threadIdx.x < nchan && threadIdx.y == 0){
		s_helper_nchan2[threadIdx.x]=s_helper_nchan3[threadIdx.x]/
				D_ERRSIN_BLOCK(__tx);
		s_helper_nchan2[threadIdx.x]=s_helper_nchan2[threadIdx.x]*s_helper_nchan2[threadIdx.x];
		d_deltatot[(nchan*blockIdx.x)+threadIdx.x]=s_helper_nchan2[threadIdx.x];
	}
	__syncthreads();

	//chi2 = deltatot#chi2totarr
	if(threadIdx.x == 0 && threadIdx.y == 0){
		s_broadcast_vars[9]=0.0;
		#pragma unroll
		for (int i = 0; i < nchan; ++i){
			s_broadcast_vars[9]+=s_helper_nchan2[i]*(1.0/nchan);
		}
	}
	__syncthreads();

}






template <typename t_precision, typename T, int nchan, int nb2> __global__ void
__launch_bounds__(192, 10) dem_block(
	//a_struc
	t_precision* d_const_nchan,
	t_precision* d_BASIS,
	t_precision* d_TR_NORM,
	t_precision* d_A_ARRAY,
	t_precision* d_A_INV,
	t_precision* d_FLAT_COFFS,
	t_precision* d_EXPTIMES,
	t_precision* d_WVEC,
	//extensions to a_struc
	t_precision* d_NORMFAC,
	t_precision* d_a2_array,
	t_precision* d_basis22,
	t_precision* d_chi2_reg_ends,
	t_precision* d_chi2_target,
	t_precision* d_chi2thold,
	T*      d_nconfs,
	t_precision* minconds,
	t_precision* d_a_inv_arr,
	t_precision* d_chi2thold_iter,
	//Helper structures in device memory
	t_precision* d_data_out,
	t_precision* d_dem_initial,
	t_precision* d_dempos,
	t_precision* d_datapos_test,
	t_precision* d_deltatot,
	t_precision* d_chi2_current,
	t_precision* d_tr_norm2,
	t_precision* d_a_inv_scaled,
	t_precision* d_sigs2_diag,
	t_precision* d_datavec,
	t_precision* d_data2vec,
	t_precision* d_dem_coofs,
	t_precision* d_dempos_out,
	t_precision* d_deltadata,
	t_precision* d_dem_noextrap,
	t_precision* d_lastdem,
	t_precision* d_dem,
	t_precision* d_dem_test,
	t_precision* d_dempos_test,
	t_precision* d_last_dem,
	//Allocate data dependent memory
	T		d_nchan_int[],
	 int32_t    chunk_size
)
{

	//Init shared memory structures based on their size (due to heavy reuse)
    __shared__ t_precision s_broadcast_vars[10];
    	//[0]=alpha
    	//[1]=chi2
    	//[2]=bisect_start
    	//[3]=alpha_low
		//[4]=alpha_high
		//[5]=break loop
    	//[6]=chi2_current
    	//[7]=chi2
    __shared__ t_precision s_helper_nb2[nb2];
    __shared__ t_precision s_helper_nchan[nchan];
    __shared__ t_precision s_helper_nchan2[nchan];
    __shared__ t_precision s_helper_nchan3[nchan];
    __shared__ t_precision s_helper_nchan_nchan[nchan*nchan];
    __shared__ t_precision s_helper_nchan_nchan2[nchan*nchan];
    __shared__ t_precision s_helper_nchan_nb2[nchan*nb2];

    t_precision r_helper = 0.0;


    if (threadIdx.x < nchan && threadIdx.y == 0){
    	D_ERRSIN_BLOCK(__tx)=(T)sqrt(MAX((D_DATAIN_BLOCK(__tx))*__D_CONST_DNPP(__tx),0.0)+__D_CONST_RDN(__tx)*__D_CONST_RDN(__tx));
    }


	// Loop over regularization chi squareds
    for (int k = 0; k < 3; ++k){

		//Regularize the data so that its chi squared (relative to the original data)
		//is equal to chi2_reg_current
    	//----------------------------------------------------------------------------------------
    	//firdem_regularize_data()

    	//Check to see if the data values are so small that zero is an acceptable solution
    	if (threadIdx.x < nchan && threadIdx.y == 0){
    		s_helper_nb2[threadIdx.x] = (t_precision)D_DATAIN_BLOCK(threadIdx.x)/D_ERRSIN_BLOCK(threadIdx.x);
    		s_helper_nb2[threadIdx.x] = s_helper_nb2[threadIdx.x]*s_helper_nb2[threadIdx.x];
    	}

    	if (threadIdx.x == 0 && threadIdx.y == 0){
			#pragma unroll
    		for (int i = 1; i < nchan; ++i){
    			s_helper_nb2[0] += s_helper_nb2[i];
    		}

    		//As only one thread is active, init bisect_start and break loop var
    		s_broadcast_vars[2]=0.0;
    		s_broadcast_vars[3]=0.0;
    		s_broadcast_vars[5]=0.0;
    	}
    	__syncthreads();
    	if(s_helper_nb2[0] < d_chi2_reg_ends[k]){
    		//return 0 as solution
    		if (threadIdx.x < nchan && threadIdx.y == 0){
    			d_data_out[blockIdx.x*nchan+threadIdx.x] = 0.0;
    			s_helper_nchan3[threadIdx.x] = 0.0;
    			s_helper_nchan[threadIdx.x] = __D_CONST_TR_NORM(__tx)*d_EXPTIMES[threadIdx.x];
    		}
    	}else{

    		//Essentially no regularization required
    		if(d_chi2_reg_ends[k]/nchan < 0.0001){
        		//return inputdata without modification
        		if (threadIdx.x < nchan && threadIdx.y == 0){
        			d_data_out[threadIdx.x+(nchan*blockIdx.x)] =
        					D_DATAIN_BLOCK(threadIdx.x);
        			s_helper_nchan3[threadIdx.x] = D_DATAIN_BLOCK(threadIdx.x);
        			s_helper_nchan[threadIdx.x] = __D_CONST_TR_NORM(__tx)*d_EXPTIMES[threadIdx.x];
        		}
    		}else{


    			// a_array is computed with normalized temperature response functions so that
    			// it has unit diagonals and off-diagonal elements less than one. The data vector
    			// must be divided by the same normalization factor to match.
    			if (threadIdx.x < nchan && threadIdx.y == 0){
    				s_helper_nchan[threadIdx.x] = __D_CONST_TR_NORM(__tx)*d_EXPTIMES[threadIdx.x];
    				d_tr_norm2[(nchan*blockIdx.x)+threadIdx.x] = s_helper_nchan[threadIdx.x];
    			}
    			__syncthreads();
    			if (threadIdx.x < nchan && threadIdx.y < nchan){
    				s_helper_nchan_nchan[threadIdx.y*nchan + threadIdx.x]=
    						d_A_INV[threadIdx.y*nchan + threadIdx.x]/
    						(s_helper_nchan[threadIdx.x]*s_helper_nchan[threadIdx.y]);
    				d_a_inv_scaled[(nchan*nchan*blockIdx.x)+threadIdx.y*nchan + threadIdx.x] =
    						s_helper_nchan_nchan[threadIdx.y*nchan + threadIdx.x];

    				d_sigs2_diag[(nchan*nchan*blockIdx.x)+threadIdx.y*nchan + threadIdx.x] = (t_precision)0.0;
    			}
    			__syncthreads();
    			if (threadIdx.x < nchan && threadIdx.y == threadIdx.x){
    				d_sigs2_diag[(nchan*nchan*blockIdx.x)+threadIdx.y*nchan + threadIdx.x] =
//    						pow(((double)1.0/d_errsin[blockIdx.x+(threadIdx.x*chunk_size)]),2);
    						((t_precision)1.0/D_ERRSIN_BLOCK(__tx))*((t_precision)1.0/D_ERRSIN_BLOCK(__tx));
    			}


				#pragma unroll
				for (int kk = 0; kk < 50; ++kk){


					if (threadIdx.x < nchan && threadIdx.y == 0){
						s_helper_nchan2[threadIdx.x] = D_DATAIN_BLOCK(__tx) /
								s_helper_nchan[threadIdx.x];
						d_datavec[(nchan*blockIdx.x) + threadIdx.x] = s_helper_nchan2[threadIdx.x];

					}
					__syncthreads();

					//Intermediate vector appearing multiple times in the calculation
					if (threadIdx.x < nchan && threadIdx.y < nchan){
						s_helper_nchan_nchan2[threadIdx.y*nchan + threadIdx.x]=
								d_A_INV[threadIdx.y*nchan + threadIdx.x]*s_helper_nchan2[threadIdx.x];
						if(k==1 && kk==0){
						}
					}
					__syncthreads();
					if (threadIdx.x < nchan && threadIdx.y == 0){
						s_helper_nchan3[threadIdx.x] = 0.0;
						#pragma unroll
			    		for (int i = 0; i < nchan; ++i){
			    			s_helper_nchan3[threadIdx.x] += s_helper_nchan_nchan2[threadIdx.x*nchan + i];
			    		}
			    		if(k==1 && kk==0){

			    		}
			    		s_helper_nchan3[threadIdx.x] =
			    				s_helper_nchan3[threadIdx.x]/s_helper_nchan[threadIdx.x];
			    		d_data2vec[(nchan*blockIdx.x)+threadIdx.x]= s_helper_nchan3[threadIdx.x];
					}


					// Multiplicative factor ensuring chi squared between the regularized data vectors
					// for this step vs. the previous step is chi20
					if (kk == 0){
						if (threadIdx.x < nchan && threadIdx.y == 0){
							s_helper_nb2[threadIdx.x] =
									s_helper_nchan3[threadIdx.x]*s_helper_nchan3[threadIdx.x] *
									(t_precision)D_ERRSIN_BLOCK(__tx)*(t_precision)D_ERRSIN_BLOCK(__tx);
							if(k==1 && kk==0){
							}
						}
				    	if (threadIdx.x == 0 && threadIdx.y == 0){
							#pragma unroll
				    		for (int i = 1; i < nchan; ++i){
				    			s_helper_nb2[0] += s_helper_nb2[i];
				    		}
				    		s_broadcast_vars[0] = sqrt(9.0*d_chi2_reg_ends[k]/s_helper_nb2[0]);
				    	}
					}
					__syncthreads();

					// The difference between the regularized data vector for this step and
					// the previous step:
					if (threadIdx.x < nchan && threadIdx.y < nchan){
						s_helper_nchan_nchan[threadIdx.y*nchan + threadIdx.x]=
							s_helper_nchan_nchan[threadIdx.y*nchan + threadIdx.x]*s_broadcast_vars[0]+
							d_sigs2_diag[(nchan*nchan*blockIdx.x)+threadIdx.y*nchan + threadIdx.x];
					}
					__syncthreads();
					Inverse<t_precision, nchan>(s_helper_nchan_nchan,s_helper_nchan_nchan2);
					__syncthreads();


					if (threadIdx.x < nchan && threadIdx.y < nchan){
						s_helper_nchan_nchan[threadIdx.y*nchan + threadIdx.x]=
								s_helper_nchan_nchan2[threadIdx.y*nchan + threadIdx.x]*
								s_helper_nchan3[threadIdx.x]*s_broadcast_vars[0];
					}
					__syncthreads();
					if (threadIdx.x < nchan && threadIdx.y == 0){
						s_helper_nchan3[threadIdx.x] = 0.0;
						#pragma unroll
			    		for (int i = 0; i < nchan; ++i){
			    			s_helper_nchan3[threadIdx.x] += s_helper_nchan_nchan[threadIdx.x*nchan + i];
			    		}
			    		s_helper_nchan3[threadIdx.x] = s_helper_nchan2[threadIdx.x]-
			    				s_helper_nchan3[threadIdx.x]/s_helper_nchan[threadIdx.x];
			    		d_datavec[(nchan*blockIdx.x)+threadIdx.x]= s_helper_nchan3[threadIdx.x];


			    		//Finish if chi squared between datavec and datavec0 reaches chi2_end:
					    s_helper_nchan2[threadIdx.x] = (((t_precision)D_DATAIN_BLOCK(__tx) -
			    				s_helper_nchan3[threadIdx.x]*s_helper_nchan[threadIdx.x])/
			    				(t_precision)D_ERRSIN_BLOCK(__tx));
					    s_helper_nchan2[threadIdx.x]= s_helper_nchan2[threadIdx.x]*s_helper_nchan2[threadIdx.x];
				    	//todo: use warp reduction for this sum
				    	if (threadIdx.x == 0 && threadIdx.y == 0){
							#pragma unroll
				    		for (int i = 1; i < nchan; ++i){
				    			s_helper_nchan2[0] += s_helper_nchan2[i];
				    		}
				    		//chi2
				    		s_broadcast_vars[1] = s_helper_nchan2[0];
				    	}
					}
					__syncthreads();

					//Do some preloading of values we need in the next iteration
					if (threadIdx.x < nchan && threadIdx.y < nchan){
						s_helper_nchan_nchan[threadIdx.y*nchan + threadIdx.x]=
							d_a_inv_scaled[(nchan*nchan*blockIdx.x)+threadIdx.y*nchan + threadIdx.x];
					}



					if (threadIdx.x == 0 && threadIdx.y == 0){

						if(s_broadcast_vars[2]==1.0){
							if(s_broadcast_vars[1]<d_chi2_reg_ends[k]){
								s_broadcast_vars[3]=s_broadcast_vars[0];
							}
							if(s_broadcast_vars[1]>d_chi2_reg_ends[k]){
								s_broadcast_vars[4]=s_broadcast_vars[0];
							}
							//alpha = alpha_low+0.5*(alpha_high-alpha_low)
							s_broadcast_vars[0]=s_broadcast_vars[3]+0.5*(
									s_broadcast_vars[4]-s_broadcast_vars[3]);
						}

						if(s_broadcast_vars[1]<d_chi2_reg_ends[k] && s_broadcast_vars[2]==0.0){
							s_broadcast_vars[0]=5*s_broadcast_vars[0];
						}

						if(s_broadcast_vars[1]>d_chi2_reg_ends[k] && s_broadcast_vars[2]==0.0){
							s_broadcast_vars[2]=1.0;
							s_broadcast_vars[4]=s_broadcast_vars[0];
							//printf("Adapted alpha to k%d B%d kk%d %.17g %.17g \n",k,blockIdx.x,kk,s_broadcast_vars[3],s_broadcast_vars[4]);
							s_broadcast_vars[0]=s_broadcast_vars[3]+0.5*(
									s_broadcast_vars[4]-s_broadcast_vars[3]);

						}

						if(abs(s_broadcast_vars[1]-d_chi2_reg_ends[k])/d_chi2_reg_ends[k] < 0.05){
							s_broadcast_vars[5]=1.0;
						}
					}
					__syncthreads();

					if(s_broadcast_vars[5]==1.0){break;}

				}

				//return,datavec*tr_norm
				if (threadIdx.x < nchan && threadIdx.y == 0){
					s_helper_nchan3[threadIdx.x]=s_helper_nchan3[threadIdx.x]*s_helper_nchan[threadIdx.x];
					d_data_out[(nchan*blockIdx.x)+ threadIdx.x]=
							s_helper_nchan3[threadIdx.x];

				}

    		}
    	}
    	//firdem_regularize_data()
    	//----------------------------------------------------------------------------------------


    	__syncthreads();
    	//Compute the first pass DEM corresponding to the regularized data
		if (threadIdx.x < nchan && threadIdx.y < nchan){
			s_helper_nchan_nchan2[threadIdx.y*nchan + threadIdx.x]=
					d_A_INV[threadIdx.y*nchan + threadIdx.x]*
					(s_helper_nchan3[threadIdx.x]/s_helper_nchan[threadIdx.x]);
		}
		__syncthreads();
		if (threadIdx.x < nchan && threadIdx.y == 0){
			s_helper_nchan[threadIdx.x] = 0.0;
    		for (int i = 0; i < nchan; ++i){
    			s_helper_nchan[threadIdx.x] += s_helper_nchan_nchan2[threadIdx.x*nchan + i];
    		}
		}
		__syncthreads();


		s_helper_nchan_nb2[threadIdx.y + threadIdx.x*nchan]=
				d_basis22[threadIdx.y + threadIdx.x*nchan]*s_helper_nchan[threadIdx.y];
		__syncthreads();
		if (threadIdx.x < nb2 && threadIdx.y == 0){
			s_helper_nb2[threadIdx.x]=0.0;
			#pragma unroll
			for (int i = 0; i < nchan; ++i){
				s_helper_nb2[threadIdx.x]+=s_helper_nchan_nb2[threadIdx.x*nchan + i];
			}
			d_dem_initial[blockIdx.x*nb2 + threadIdx.x]=s_helper_nb2[threadIdx.x];

			if(s_helper_nb2[threadIdx.x]<0){s_helper_nb2[threadIdx.x]=0.0;}
		}
		__syncthreads();


		//datapos_test(*) = a2_array#dempos_test
		s_helper_nchan_nb2[threadIdx.x + threadIdx.y*nb2]=
				d_a2_array[threadIdx.x + threadIdx.y*nb2]*s_helper_nb2[threadIdx.x];
		__syncthreads();
		if (threadIdx.x < nchan && threadIdx.y == 0){
			s_helper_nchan[threadIdx.x]=0.0;
			#pragma unroll
			for (int i = 0; i < nb2; ++i){
				s_helper_nchan[threadIdx.x]+=s_helper_nchan_nb2[threadIdx.x*nb2 + i];
			}
		}

		__syncthreads();
		//deltatot = reform( ( (datapos_test-datain(i,j,*)) / errsin(i,j,*) )^2)
		if (threadIdx.x < nchan && threadIdx.y == 0){
			s_helper_nchan[threadIdx.x]=(s_helper_nchan[threadIdx.x]-
					(t_precision)D_DATAIN_BLOCK(__tx) )/D_ERRSIN_BLOCK(__tx);
			s_helper_nchan[threadIdx.x]=s_helper_nchan[threadIdx.x]*s_helper_nchan[threadIdx.x];

		}
		__syncthreads();
		//chi2_current = total(deltatot/nchan)
		if (threadIdx.x == 0 && threadIdx.y == 0){
			s_broadcast_vars[6]=0.0;
			#pragma unroll
			for(int i = 0; i < nchan; ++i){
				s_broadcast_vars[6]+=s_helper_nchan[i]/(t_precision)nchan;
			}
			if (s_broadcast_vars[6] <= *d_chi2thold){
				s_broadcast_vars[5]=1.0;
			}else{
				s_broadcast_vars[5]=0.0;
			}
		}
		__syncthreads();

		if (s_broadcast_vars[5]==1.0){
			if (threadIdx.y == 0 && threadIdx.x < nb2){
				d_dem_coofs[blockIdx.x*(nb2+3)+threadIdx.x]=s_helper_nb2[threadIdx.x];
				if(d_dem_coofs[blockIdx.x*(nb2+3)+threadIdx.x]<0){d_dem_coofs[blockIdx.x*(nb2+3)+threadIdx.x]=0;}
				if(threadIdx.y == 0 && threadIdx.x == 0){
					s_broadcast_vars[7]=s_broadcast_vars[6];
					d_dem_coofs[blockIdx.x*(nb2+3)+nb2]=s_broadcast_vars[6];
				}
			}
			break;
		}
		__syncthreads();

		//Attempt to iterate away negative emission in the first pass DEM
		firdem_iterate<t_precision, T, nchan, nb2>(
				s_broadcast_vars,
				s_helper_nb2,
				s_helper_nchan,
				s_helper_nchan2,
				s_helper_nchan3,
				s_helper_nchan_nchan,
				s_helper_nchan_nchan2,
				s_helper_nchan_nb2,
				k,
				minconds,
				d_WVEC,
				d_tr_norm2,
				d_dempos,
				d_dempos_out,
				d_a_inv_scaled,
				d_datapos_test,
				d_a2_array,
				d_deltadata,
				d_nchan_int,
				d_deltatot,
				chunk_size,
				d_basis22,
				d_dem_initial,
				d_data_out,
				d_dem_noextrap,
				d_lastdem,
				d_dem,
				d_dem_test,
				d_dempos_test,
				d_last_dem,
				d_a_inv_arr,
				d_chi2thold_iter
		);

		__syncthreads();

		// Compute chi squared relative to the original data

		//deltatot = reform(((datapos-datain(i,j,*))/errsin(i,j,*))^2)
		//s_helper_nchan = datapos
		if (threadIdx.x < nchan && threadIdx.y == 0){
			s_helper_nchan[threadIdx.x]=(s_helper_nchan[threadIdx.x]-
					(t_precision)D_DATAIN_BLOCK(__tx) )/D_ERRSIN_BLOCK(__tx) ;
			s_helper_nchan[threadIdx.x]=(s_helper_nchan[threadIdx.x]*s_helper_nchan[threadIdx.x])/nchan;
		}
		__syncthreads();
		//chi2_current = total(deltatot)
		if (threadIdx.x == 0 && threadIdx.y == 0){
			s_broadcast_vars[6]=0.0;
			#pragma unroll
			for(int i = 0; i < nchan; ++i){
				s_broadcast_vars[6]+=s_helper_nchan[i];
			}
			//if(k eq 0) then chi2=chi2_current
			if(k==0){s_broadcast_vars[7]=s_broadcast_vars[6];}
		}
		__syncthreads();

		//Update the DEM if the new chi squared is better than the previous best
		if (threadIdx.x <nb2 && threadIdx.y == 0){
			if(k == 0 || s_broadcast_vars[6]<s_broadcast_vars[7]){
				d_dem_coofs[blockIdx.x*(nb2+3)+threadIdx.x]=d_dem[blockIdx.x*nb2 + threadIdx.x];
				if(d_dem_coofs[blockIdx.x*(nb2+3)+threadIdx.x]<0){d_dem_coofs[blockIdx.x*(nb2+3)+threadIdx.x]=0;}
				if(threadIdx.y == 0 && threadIdx.x == 0){
					s_broadcast_vars[7]=s_broadcast_vars[6];
					d_dem_coofs[blockIdx.x*(nb2+3)+nb2]=s_broadcast_vars[6];
				}
			}
		}
		__syncthreads();

		if(s_broadcast_vars[7]<=*d_chi2thold){break;}

    }
    __syncthreads();

}











template <typename t_precision, typename t_input, size_t t_nb2, size_t t_nchan, size_t t_streams>
void Dem<t_precision,t_input,t_nb2,t_nchan,t_streams>::preprocessing(){


	 // We load some static AIA information calculated by IDL as constants
	 fillAIAConstants(aia_constants);
	 // therefore skip firstpass()

	 //Number of temperature grid points
	 size_t nt=(t_nb2+1)*10+1;

	 //calculate the ainv_array for each mincond-iteration
	 t_precision wp[t_nchan][t_nchan];
	 t_precision tmp[t_nchan][t_nchan];

	 for(size_t m=0;m<3;m++){
		for(size_t i=0;i<t_nchan;i++){
			for(size_t j=0;j<t_nchan;j++){wp[i][j]=0.0;}
			wp[i][i] = (aia_constants.WVEC[i]>= aia_constants.minconds[m]*aia_constants.WVEC[0]) ? 1.0/aia_constants.WVEC[i]:0.0;
		}

		for(size_t i = 0; i < t_nchan; ++i){
			for(size_t j = 0; j < t_nchan; ++j){
				tmp[i][j] = 0.0;
				for(size_t k = 0; k < t_nchan; ++k){
					tmp[i][j] += aia_constants.V[i][k] * wp[k][j];
				}
				aia_constants.A_INV_ARR[m][i][j] = 0.0;
			}
		}
		for(size_t i = 0; i < t_nchan; ++i){
			for(size_t j = 0; j < t_nchan; ++j){
				for(size_t k = 0; k < t_nchan; ++k){
					aia_constants.A_INV_ARR[m][i][j] += tmp[i][k] * aia_constants.U[k][j];
				}
			}
		}
	 }



}



template <typename t_precision, typename t_input, size_t t_nb2, size_t t_nchan, size_t t_streams>
Dem<t_precision,t_input,t_nb2,t_nchan,t_streams>::Dem(int64_t chunk_size_in, bool EMWMT) :
GPUHandler<t_streams>(){


	// increase CUDA HeapSize to 256MB
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, 256000000);
	cudaCheckErrors("Increase cuda heap size.");


	// the chunk_size will be the grid size of the kernel lunch
	if (chunk_size_in > 16384) {
		chunk_size = 16384;
		nbrPasses = chunk_size_in / chunk_size;
		last_chunk_size = chunk_size_in - (nbrPasses*chunk_size);
		if(last_chunk_size == 0){
			last_chunk_size = chunk_size;
			nbrPasses -= 1;
		}

	}else{
		nbrPasses = 0;
		last_chunk_size= chunk_size_in;
		chunk_size = chunk_size_in;
	}
	calc_EMWMT = EMWMT;

	// do AIA specific preprocessing
	preprocessing();

	// allocate GPU memory in large blocks
	cudaMalloc((void**) &d_const_nchan		, sizeof(t_precision)*__SIZE_D_CONST_NCHAN);

	// copy values to GPU memory
	cudaMemcpyAsync(__D_CONST_DNPP_ADDR		, &aia_constants.dnpp		, sizeof(t_precision)*t_nchan			, cudaMemcpyHostToDevice, this->getStream(0));
	cudaMemcpyAsync(__D_CONST_RDN_ADDR		, &aia_constants.rdn		, sizeof(t_precision)*t_nchan			, cudaMemcpyHostToDevice, this->getStream(0));
	cudaMemcpyAsync(__D_CONST_TR_NORM_ADDR	, &aia_constants.TR_NORM	, sizeof(t_precision)*t_nchan			, cudaMemcpyHostToDevice, this->getStream(0));

	// allocate constant GPU memory
		//a_struc
		cudaMalloc((void**) &d_V              , sizeof(t_precision)*t_nchan*t_nchan  );
		cudaMalloc((void**) &d_U              , sizeof(t_precision)*t_nchan*t_nchan  );
		cudaMalloc((void**) &d_WP             , sizeof(t_precision)*t_nchan*t_nchan  );
		cudaMalloc((void**) &d_BASIS          , sizeof(t_precision)*t_nb2*331        );
		cudaMalloc((void**) &d_TR_NORM        , sizeof(t_precision)*t_nchan        );
		cudaMalloc((void**) &d_A_ARRAY        , sizeof(t_precision)*t_nchan*t_nchan  );
		cudaMalloc((void**) &d_A_INV          , sizeof(t_precision)*t_nchan*t_nchan  );
		cudaMalloc((void**) &d_FLAT_COFFS     , sizeof(t_precision)*t_nchan        );
		cudaMalloc((void**) &d_EXPTIMES       , sizeof(t_precision)*t_nchan        );
		cudaMalloc((void**) &d_WVEC           , sizeof(t_precision)*t_nchan        );
		//extensions to a_struc
		cudaMalloc((void**) &d_NORMFAC        , sizeof(t_precision)*t_nchan        );
		cudaMalloc((void**) &d_a2_array       , sizeof(t_precision)*t_nchan*t_nb2    );
		cudaMalloc((void**) &d_basis22        , sizeof(t_precision)*t_nb2*t_nchan    );
		cudaMalloc((void**) &d_chi2_reg_ends  , sizeof(t_precision)*3            );
		cudaMalloc((void**) &d_chi2_target    , sizeof(t_precision)              );
		cudaMalloc((void**) &d_chi2thold      , sizeof(t_precision)              );
		cudaMalloc((void**) &d_nconfs         , sizeof(t_input)             );
		cudaMalloc((void**) &d_minconds       , sizeof(t_precision)*3            );
		cudaMalloc((void**) &d_a_inv_arr      , sizeof(t_precision)*3*t_nchan*t_nchan);
		cudaMalloc((void**) &d_chi2thold_iter , sizeof(t_precision)*3            );
		cudaMalloc((void**) &d_logt           , sizeof(t_precision)*331            );
		cudaCheckErrors("cudaMalloc: AIA Constants");

	// copy values of constant GPU memory
		//a_struc
		cudaMemcpyAsync(d_BASIS          , &aia_constants.BASIS          ,sizeof(t_precision)*t_nb2*331        , cudaMemcpyHostToDevice, this->getStream(0));
		cudaMemcpyAsync(d_TR_NORM        , &aia_constants.TR_NORM        ,sizeof(t_precision)*t_nchan        , cudaMemcpyHostToDevice, this->getStream(0));
		cudaMemcpyAsync(d_A_ARRAY        , &aia_constants.A_ARRAY        ,sizeof(t_precision)*t_nchan*t_nchan  , cudaMemcpyHostToDevice, this->getStream(0));
		cudaMemcpyAsync(d_A_INV          , &aia_constants.A_INV          ,sizeof(t_precision)*t_nchan*t_nchan  , cudaMemcpyHostToDevice, this->getStream(0));
		cudaMemcpyAsync(d_FLAT_COFFS     , &aia_constants.FLAT_COFFS     ,sizeof(t_precision)*t_nchan        , cudaMemcpyHostToDevice, this->getStream(0));
		cudaMemcpyAsync(d_EXPTIMES       , &aia_constants.EXPTIMES       ,sizeof(t_precision)*t_nchan        , cudaMemcpyHostToDevice, this->getStream(0));
		cudaMemcpyAsync(d_WVEC           , &aia_constants.WVEC           ,sizeof(t_precision)*t_nchan        , cudaMemcpyHostToDevice, this->getStream(0));
		//extensions to a_struc
		cudaMemcpyAsync(d_NORMFAC        , &aia_constants.NORMFAC        ,sizeof(t_precision)*t_nchan        , cudaMemcpyHostToDevice, this->getStream(0));
		cudaMemcpyAsync(d_a2_array       , &aia_constants.a2_array       ,sizeof(t_precision)*t_nchan*t_nb2    , cudaMemcpyHostToDevice, this->getStream(0));
		cudaMemcpyAsync(d_basis22        , &aia_constants.basis22        ,sizeof(t_precision)*t_nb2*t_nchan    , cudaMemcpyHostToDevice, this->getStream(0));
		cudaMemcpyAsync(d_chi2_reg_ends  , &aia_constants.chi2_reg_ends  ,sizeof(t_precision)*3            , cudaMemcpyHostToDevice, this->getStream(0));
		cudaMemcpyAsync(d_chi2_target    , &aia_constants.chi2_target    ,sizeof(t_precision)              , cudaMemcpyHostToDevice, this->getStream(0));
		cudaMemcpyAsync(d_chi2thold      , &aia_constants.chi2thold      ,sizeof(t_precision)              , cudaMemcpyHostToDevice, this->getStream(0));
		cudaMemcpyAsync(d_nconfs         , &aia_constants.nconfs         ,sizeof(t_input)             , cudaMemcpyHostToDevice, this->getStream(0));
		cudaMemcpyAsync(d_minconds       , &aia_constants.minconds       ,sizeof(t_precision)*3            , cudaMemcpyHostToDevice, this->getStream(0));
		cudaMemcpyAsync(d_a_inv_arr      , &aia_constants.A_INV_ARR      ,sizeof(t_precision)*3*t_nchan*t_nchan, cudaMemcpyHostToDevice, this->getStream(0));
		cudaMemcpyAsync(d_chi2thold_iter , &aia_constants.chi2thold_iter ,sizeof(t_precision)*3            , cudaMemcpyHostToDevice, this->getStream(0));
		cudaMemcpyAsync(d_logt           , &aia_constants.logt           ,sizeof(t_precision)*331           , cudaMemcpyHostToDevice, this->getStream(0));
		cudaCheckErrors("cudaMemcpyAsync: AIA Constants");


		// allocate GPU memory for each stream
		for (size_t i=0; i < t_streams; i++){

				// allocate helper variables on GPU memory
				cudaMalloc((void**) &d_data_out[i]                 , sizeof(t_precision)*chunk_size*t_nchan  );
				cudaMalloc((void**) &d_dem_initial[i]              , sizeof(t_precision)*chunk_size*t_nb2    );
				cudaMalloc((void**) &d_dempos[i]                   , sizeof(t_precision)*chunk_size*t_nb2    );
				cudaMalloc((void**) &d_datapos_test[i]             , sizeof(t_precision)*chunk_size*t_nchan  );
				cudaMalloc((void**) &d_deltatot[i]                 , sizeof(t_precision)*chunk_size*t_nchan  );
				cudaMalloc((void**) &d_chi2_current[i]             , sizeof(t_precision)*chunk_size        );
				cudaMalloc((void**) &d_tr_norm[i]                  , sizeof(t_precision)*chunk_size*t_nchan      );
				cudaMalloc((void**) &d_a_inv_scaled[i]             , sizeof(t_precision)*chunk_size*t_nchan*t_nchan);
				cudaMalloc((void**) &d_sigs2_diag[i]               , sizeof(t_precision)*chunk_size*t_nchan*t_nchan);
				cudaMalloc((void**) &d_datavec[i]                  , sizeof(t_precision)*chunk_size*t_nchan      );
				cudaMalloc((void**) &d_data2vec[i]                 , sizeof(t_precision)*chunk_size*t_nchan      );
				cudaMalloc((void**) &d_dem_coffs[i]                , sizeof(t_precision)*chunk_size*(t_nb2+3)    );
				cudaMalloc((void**) &d_dempos_out[i]               , sizeof(t_precision)*chunk_size*t_nb2        );
				cudaMalloc((void**) &d_deltadata[i]                , sizeof(t_precision)*chunk_size*t_nchan      );
				cudaMalloc((void**) &d_dem_noextrap[i]             , sizeof(t_precision)*chunk_size*t_nb2        );
				cudaMalloc((void**) &d_lastdem[i]                  , sizeof(t_precision)*chunk_size*t_nb2        );
				cudaMalloc((void**) &d_dem[i]                      , sizeof(t_precision)*chunk_size*t_nb2        );
				cudaMalloc((void**) &d_dem_test[i]                 , sizeof(t_precision)*chunk_size*t_nb2        );
				cudaMalloc((void**) &d_dempos_test[i]              , sizeof(t_precision)*chunk_size*t_nb2        );
				cudaMalloc((void**) &d_last_dem[i]                 , sizeof(t_precision)*chunk_size*t_nb2        );
				cudaCheckErrors("cudaMalloc: DEM helper structures");

				// allocate space for input data on GPU memory
				cudaMalloc((void**) &d_nchan_int[i]              , sizeof(t_input)     * SIZE_D_NCHAN_INT_BLOCK   );
				cudaMallocHost((void**)&errsin[i], sizeof(t_input)*chunk_size*t_nchan); // host pinned
				cudaCheckErrors("cudaMalloc: space for input data");
		}


}


template <typename t_precision, typename t_input, size_t t_nb2, size_t t_nchan, size_t t_streams>
Dem<t_precision,t_input,t_nb2,t_nchan,t_streams>::~Dem(){


	//free const memory
	cudaFree(d_const_nchan			);

	// free memory on device
	//a_struc
	cudaFree(d_V              );
	cudaFree(d_U              );
	cudaFree(d_WP             );
	cudaFree(d_TR_NORM        );
	cudaFree(d_A_ARRAY        );
	cudaFree(d_A_INV          );
	cudaFree(d_FLAT_COFFS     );
	cudaFree(d_EXPTIMES       );
	cudaFree(d_WVEC           );
	//extensions to a_struc
	cudaFree(d_NORMFAC        );
	cudaFree(d_a2_array       );
	cudaFree(d_basis22        );
	cudaFree(d_chi2_reg_ends  );
	cudaFree(d_chi2_target    );
	cudaFree(d_chi2thold      );
	cudaFree(d_nconfs         );
	cudaFree(d_minconds       );
	cudaFree(d_a_inv_arr      );
	cudaFree(d_chi2thold_iter );
	cudaFree(d_logt           );
	cudaCheckErrors("cudaFree constants");

	for (size_t i=0; i < t_streams; i++){
		 //Helper structures in device memory
		 cudaFree(d_data_out[i]       );
		 cudaFree(d_dem_initial[i]    );
		 cudaFree(d_dempos[i]         );
		 cudaFree(d_datapos_test[i]   );
		 cudaFree(d_deltatot[i]       );
		 cudaFree(d_chi2_current[i]   );
		 cudaFree(d_tr_norm[i]        );
		 cudaFree(d_a_inv_scaled[i]   );
		 cudaFree(d_sigs2_diag[i]     );
		 cudaFree(d_datavec[i]        );
		 cudaFree(d_data2vec[i]       );
		 cudaFree(d_dem_coffs[i]      );
		 cudaFree(d_dempos_out[i]     );
		 cudaFree(d_deltadata[i]      );
		 cudaFree(d_dem_noextrap[i]   );
		 cudaFree(d_lastdem[i]        );
		 cudaFree(d_dem[i]            );
		 cudaFree(d_dem_test[i]       );
		 cudaFree(d_dempos_test[i]    );
		 cudaFree(d_last_dem[i]       );
		 //data dependent memory
		 cudaFree(d_nchan_int[i]      );
		 cudaFreeHost(errsin[i]       );
	}
	cudaCheckErrors("cudaFree done");
	LOG4CXX_INFO(logger,"firdem object deleted.");


}


template <typename t_precision, typename t_input, size_t t_nb2, size_t t_nchan, size_t t_streams>
void Dem<t_precision,t_input,t_nb2,t_nchan,t_streams>::runKernel(t_input datain[],t_precision coffs[], size_t stream){

	// in case the chunk size is to big, pipeline the work
	for (int64_t tile_nr = 0;tile_nr < nbrPasses; tile_nr++){

		LOG4CXX_INFO(logger,"Tiling chunk as too big for GPU memory.");

		 // copy input values for kernel
		 cudaMemcpyAsync(D_DATAIN_ADDR_BLOCK(stream)   , &datain[tile_nr*chunk_size*t_nchan]  ,sizeof(t_input)*t_nchan*chunk_size  , cudaMemcpyHostToDevice, this->getStream(stream));
		 cudaCheckErrors("cudaMemcpyAsync: kernel input data");

		 // Setup execution parameters
		 dim3 threads(32, 6);
		 dim3 grid(chunk_size);

		 //Run kernel
			 dem_block<t_precision,t_input,6,32><<< grid, threads , 0, this->getStream(stream)>>>(
			 	//a_struc
			 	d_const_nchan,
			 	d_BASIS,
			 	d_TR_NORM,
			 	d_A_ARRAY,
			 	d_A_INV,
	//		 	d_TR,
			 	d_FLAT_COFFS,
			 	d_EXPTIMES,
			 	d_WVEC,
			 	//extensions to a_struc
			 	d_NORMFAC,
			 	d_a2_array,
			 	d_basis22,
			 	d_chi2_reg_ends,
			 	d_chi2_target,
			 	d_chi2thold,
			 	d_nconfs,
			 	d_minconds,
			 	d_a_inv_arr,
			 	d_chi2thold_iter,
			 	//Helper structures in device memory
			 	d_data_out[stream],
			 	d_dem_initial[stream],
			 	d_dempos[stream],
			 	d_datapos_test[stream],
			 	d_deltatot[stream],
			 	d_chi2_current[stream],
			 	d_tr_norm[stream],
			 	d_a_inv_scaled[stream],
			 	d_sigs2_diag[stream],
			 	d_datavec[stream],
			 	d_data2vec[stream],
			 	d_dem_coffs[stream],
			 	d_dempos_out[stream],
			 	d_deltadata[stream],
			 	d_dem_noextrap[stream],
			 	d_lastdem[stream],
			 	d_dem[stream],
				d_dem_test[stream],
				d_dempos_test[stream],
				d_last_dem[stream],
			 	//Allocate data dependent memory
			 	d_nchan_int[stream],
			 	(int32_t)chunk_size
			 );
			 cudaCheckErrors("kernel launch");

			 // calculate EMWMT and EMTOT
			 if(calc_EMWMT){
				 firdem_emwtemp<t_precision,t_input,6,32><<< grid, threads , 0, this->getStream(stream)>>>(
					d_BASIS,d_logt,d_dem_coffs[stream]);
				 cudaCheckErrors("kernel launch firdem_emwtemp");
			 }

			 cudaMemcpyAsync(&coffs[tile_nr*chunk_size*(t_nb2+3)] , d_dem_coffs[stream]   ,sizeof(t_precision)*(t_nb2+3)*chunk_size  , cudaMemcpyDeviceToHost, this->getStream(stream));
			 cudaCheckErrors("cudaMemcpyAsync: result copy job");

	}


	 // copy input values for kernel
	 cudaMemcpyAsync(D_DATAIN_ADDR_BLOCK(stream)   , &datain[nbrPasses*chunk_size*t_nchan] ,sizeof(t_input)*t_nchan*last_chunk_size  , cudaMemcpyHostToDevice, this->getStream(stream));
	 cudaCheckErrors("cudaMemcpyAsync: kernel input data");

	 // Setup execution parameters
	 dim3 threads(32, 6);
	 dim3 grid(last_chunk_size);

	 //Run kernel
		 dem_block<t_precision,t_input,6,32><<< grid, threads , 0, this->getStream(stream)>>>(
		 	//a_struc
		 	d_const_nchan,
		 	d_BASIS,
		 	d_TR_NORM,
		 	d_A_ARRAY,
		 	d_A_INV,
//		 	d_TR,
		 	d_FLAT_COFFS,
		 	d_EXPTIMES,
		 	d_WVEC,
		 	//extensions to a_struc
		 	d_NORMFAC,
		 	d_a2_array,
		 	d_basis22,
		 	d_chi2_reg_ends,
		 	d_chi2_target,
		 	d_chi2thold,
		 	d_nconfs,
		 	d_minconds,
		 	d_a_inv_arr,
		 	d_chi2thold_iter,
		 	//Helper structures in device memory
		 	d_data_out[stream],
		 	d_dem_initial[stream],
		 	d_dempos[stream],
		 	d_datapos_test[stream],
		 	d_deltatot[stream],
		 	d_chi2_current[stream],
		 	d_tr_norm[stream],
		 	d_a_inv_scaled[stream],
		 	d_sigs2_diag[stream],
		 	d_datavec[stream],
		 	d_data2vec[stream],
		 	d_dem_coffs[stream],
		 	d_dempos_out[stream],
		 	d_deltadata[stream],
		 	d_dem_noextrap[stream],
		 	d_lastdem[stream],
		 	d_dem[stream],
			d_dem_test[stream],
			d_dempos_test[stream],
			d_last_dem[stream],
		 	//Allocate data dependent memory
		 	d_nchan_int[stream],
		 	(int32_t)chunk_size
		 );
		 cudaCheckErrors("kernel launch");

		 // calculate EMWMT and EMTOT
		 if(calc_EMWMT){
			 firdem_emwtemp<t_precision,t_input,6,32><<< grid, threads , 0, this->getStream(stream)>>>(
				d_BASIS,d_logt,d_dem_coffs[stream]);
			 cudaCheckErrors("kernel launch firdem_emwtemp");
		 }

		 //cudaMemcpyAsync(&coffs[nbrPasses*chunk_size*t_nb2] , d_dem_coffs[stream]   ,sizeof(t_precision)*32*last_chunk_size  , cudaMemcpyDeviceToHost, streams[stream]);
		 cudaMemcpyAsync(&coffs[nbrPasses*chunk_size*(t_nb2+3)] , d_dem_coffs[stream]   ,sizeof(t_precision)*(t_nb2+3)*last_chunk_size  , cudaMemcpyDeviceToHost, this->getStream(stream));
		 cudaCheckErrors("cudaMemcpyAsync: result copy job");

		 // record an syncEvent to Query the status of the kernel
		 this->setSyncEvent(stream);

}


// init the template classes as nvcc doesn't know the used types
template class Dem<double,int16_t,32,6,3>;
template class Dem<double,int16_t,32,6,2>;
template class Dem<double,int16_t,32,6,1>;
