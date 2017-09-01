/**
 * @file firdem.h
 *
 * @author simon.marcin
 *
 * @brief definition of DEM algorithm class uses GPUHandler as base class.
 *
 * @tparam t_streams number of used GPU streams
 * @tparam t_precision type of output
 * @tparam t_input type of input
 * @tparam t_nb2 number of used coefficients (coarse temp bins)
 * @tparam t_nchan number of used AIA channels
 *
 */

#include <cuda_runtime.h>
#include <log4cxx/logger.h>
#include "AIAConstants.h"
#include "../../utils/GPUHandler.h"


template <typename t_precision, typename t_input, size_t t_nb2, size_t t_nchan, size_t t_streams>
class Dem: public GPUHandler<t_streams>{

private:

	// array of streams and events
	//cudaStream_t streams[t_streams];
    //cudaEvent_t syncEvents[t_streams];

    // the chunksize represents the gridf size of cuda blocks
    int64_t chunk_size;
    int64_t nbrPasses;
    int64_t last_chunk_size;

	// preprocessing data
	AIAConstants aia_constants;
	t_precision chi2_iter_thold ;
	bool calc_EMWMT;
	//double* dem_coffs = new double[2048*32];


	// allocate device memory which is constant
	t_precision * d_nchan[t_streams]; 	// access happens in nchan chunks
	t_precision * d_const_nchan;		// access happens in nchan chunks
	t_precision * d_nb2[t_streams];		// access happens in nb2 chunks
	t_input * d_nchan_int[t_streams];	// access happens in nchan chunks


	 // Allocate memory on device for AIAConstants

	//a_struc
	t_precision * d_V;
	t_precision * d_U;
	t_precision * d_WP;
	t_precision  * d_T;
	t_precision * d_BASIS;
	t_precision * d_TR_NORM;
	t_precision * d_A_ARRAY;
	t_precision * d_A_INV;
	t_precision * d_TR;
	t_precision * d_FLAT_COFFS;
	t_precision * d_EXPTIMES;
	t_precision * d_WVEC;
	//extensions to a_struc
	t_precision * d_NORMFAC;
	t_precision * d_a2_array;
	t_precision * d_basis22;
	t_precision * d_chi2_reg_ends;
	t_precision * d_chi2_target;
	t_precision * d_chi2thold;
	t_precision * d_chi2thold_iter;
	int16_t* d_nconfs;
	t_precision * d_minconds;
	t_precision * d_a_inv_arr;
	t_precision * d_logt;

	//Helper structures in device memory - each stream has its own data
	t_precision * d_data_out[t_streams];
	t_precision * d_dem_initial[t_streams];
	t_precision * d_dempos[t_streams];
	t_precision * d_datapos_test[t_streams];
	t_precision * d_deltatot[t_streams];
	t_precision * d_chi2_current[t_streams];
	t_precision * d_tr_norm[t_streams];
	t_precision * d_a_inv_scaled[t_streams];
	t_precision * d_sigs2_diag[t_streams];
	t_precision * d_datavec[t_streams];
	t_precision * d_data2vec[t_streams];
	t_precision * d_dem_coffs[t_streams];
	t_precision * d_dempos_out[t_streams];
	t_precision * d_deltadata[t_streams];
	t_precision * d_dem_noextrap[t_streams];
	t_precision * d_lastdem[t_streams];
	t_precision * d_dem[t_streams];
	t_precision * d_dem_test[t_streams];
	t_precision * d_dempos_test[t_streams];
	t_precision * d_last_dem[t_streams];

	t_input * d_datain[t_streams];
	t_input * d_errsin[t_streams];
	t_input * errsin[t_streams];


	void preprocessing();


public:

	Dem(int64_t chunk_size, bool EMWMT);
	~Dem();
	void runKernel(t_input datain[],t_precision coffs[], size_t stream);

};
