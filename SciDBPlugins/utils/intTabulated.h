/**
 * @file intTabluated.h
 *
 * @author simon.marcin
 *
 * @brief ...
 *
 * @tparam t_precision data type (float or double) of input and output matrix.
 *
 * @param X       The tabulated X-value data.
 * @param F       The tabulated F-value data.
 * @param size    Size of the arrays
 *
 */

#include <vector>
#include <stdio.h>
#include <stdlib.h>

// Adapted version of Numerical Recipes in C: The Art of Scientific Computing,
// published by Cambridge University Press.
template <typename t_precision>
void spline(t_precision x[], t_precision y[], size_t n, t_precision y2[]){

	t_precision p,qn,sig,un;
	std::vector<t_precision> u(n-1);

	y2[0] = u[0] = 0.0;

	for(size_t i = 1; i < n-1; i++){
		sig = (x[i] - x[i-1])/(x[i+1] - x[i-1]);
		p = sig*y2[i-1] + 2.0;
		y2[i] = (sig - 1.0)/p;
		u[i] = (y[i+1] - y[i])/(x[i+1] - x[i]) - (y[i] - y[i-1])/(x[i] - x[i-1]);
		u[i] = (6.0*u[i]/(x[i+1] - x[i-1]) - sig*u[i-1])/p;
	}

	qn = un = 0.0;

	y2[n-1] = (un - qn*u[n-2])/(qn*y2[n-2] + 1.0);
	for(int k = n-2; k >= 0; k--){
		y2[k] = y2[k]*y2[k+1] + u[k];
	}

}

// Adapted version of Numerical Recipes in C: The Art of Scientific Computing,
// published by Cambridge University Press.
template <typename t_precision>
t_precision splint(t_precision xa[], t_precision ya[], t_precision y2a[], size_t n, t_precision x){

	int klo,khi,k;
	t_precision h,b,a;
	static int	pklo=0,pkhi=1;

	  if(xa[pklo] <= x && xa[pkhi] > x){
	    klo = pklo;
	    khi = pkhi;
	  }
	  else{
	    klo = 0;
	    khi = n - 1;
	    while(khi - klo > 1){
	      k = (khi + klo) >> 1;
	      if(xa[k] > x) khi = k;
	      else          klo = k;
	    }
	    pklo = klo;
	    pkhi = khi;
	  }

	  h = xa[khi] - xa[klo];
	  if(h == 0){
	    fprintf(stderr,"-E- %s line %d: Bad xa input to function splint()\n",
	            __FILE__,__LINE__);
	    exit(EXIT_FAILURE);
	  }
	  a = (xa[khi] - x)/h;
	  b = (x - xa[klo])/h;
	  return a*ya[klo] + b*ya[khi] +
	       ((a*a*a - a)*y2a[klo] + (b*b*b - b)*y2a[khi])*(h*h)/6.0;

}

 

template <typename t_precision>
t_precision int_tabulated(t_precision X[], t_precision F[], size_t size){

	// number of segments
	size_t xsegments = size + 1;
	while(xsegments % 4 != 0){
		xsegments+=1;
	}

	// init varaibles
	size_t ii = 0;
	t_precision res = 0.0;
	t_precision* z = new t_precision[xsegments+1];
	t_precision* y2 = new t_precision[size];


	t_precision xmin = X[0];
	t_precision xmax = X[size -1];
	for(size_t i = 0; i < size; i++){
		if(X[i] < xmin){xmin = X[i];}
		if(X[i] > xmax){xmax = X[i];}
	}

	// Uniform step size
	t_precision h = (xmax-xmin) / xsegments;

	// Init spline
	spline<t_precision>(X, F, size, y2);

	// Compute the interpolates at Xgrid
	for(t_precision i = 0; i < xsegments+1; i++){
		z[(int)i]=splint<t_precision>(X, F, y2, size, (t_precision)h*i+xmin);
	}

	// Compute the integral using the 5-point Newton-Cotes formula
	for(size_t i = 0; i < (xsegments/4); i++){
		ii = (i + 1) * 4;
		res += (t_precision)2 * h * ((t_precision)7 * (z[ii-4] + z[ii]) + (t_precision)32 * (z[ii-3] + z[ii-1]) + (t_precision)12 * z[ii-2]) / (t_precision)45;
	}

	delete[] z;
	delete[] y2;

	return res;
 }
