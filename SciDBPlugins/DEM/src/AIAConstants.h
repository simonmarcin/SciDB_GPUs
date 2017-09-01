/*

 * AIAConstants.h
 *
 *  Created on: Jun 2, 2017
 *      Author: simon
 */
#pragma once
#include <stdint.h>

// define values which we get from specific AIA functions in IDL
struct AIAConstants{

	// Photon to DN conversion
	double dnpp[6] = {1.5230000019073486,1.1679999828338623,1.0240000486373901,
			0.94599997997283936,0.59600001573562622,2.1280000209808350};

	// Read noise constants
	double rdn[6] = {1.1799999475479126,1.1499999761581421,1.2000000476837158,
			1.2000000476837158,1.1799999475479126,1.1399999856948853};

	//a_struc
		double V[6][6];
		double U[6][6];
		double WP[6][6];
		float T[331];
		double BASIS[331][32];
		double TR_NORM[6];
		double A_ARRAY[6][6];
		double A_INV[6][6];
		double TR[331][6];
		double FLAT_COFFS[6];
		double EXPTIMES[6];
		double WVEC[6];

		//extensions to a_struc
		double NORMFAC[6];
		double logt[331];
		double a2_array[6][32];
		double basis22[32][6];
		double chi2_reg_ends[3];
		double chi2_target;
		double chi2thold;
		int16_t nconfs;
		double chi2_reg_probs[3] = {0.9,0.5,0.01};
		double minconds[3] = {0.01,0.05,0.1};
		double A_INV_ARR[3][6][6];
		double chi2thold_iter[3] = {1.7312430540720622,1.7312430540720622,1.7312430540720622};

};

void fillAIAConstants(AIAConstants &tofill);
