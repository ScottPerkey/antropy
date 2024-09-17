//fftw type shit wrapper
#include <fftw3.h>
void fftw_wrapper( const double *input, double*output, int size      ){

	fftw_plan plan = fftw_plan_r2r_1d( size, input, output, FFTW_R2HC, FFTW_ESTIMATE  );
	fftw_execute(plan);
	fftw_destroy_plan(plan);

}

