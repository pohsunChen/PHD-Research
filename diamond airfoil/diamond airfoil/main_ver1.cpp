/*
Program for 
simulation of inviscid flow in non-othogonal mesh
2019/02/19
P.-H. Chen
*/

#define _USE_MATH_DEFINES
#include<iostream>
#include<Windows.h>
#include<iomanip>
#include<omp.h>
#include<cmath>
#include<fstream>
#include<time.h>

using namespace std;

// geometry parameter
double Lx = 20.0;
double Ly = 5.0;
int Nx_per_unit = 16;
int Ny_per_unit = 32;
int fine_para = 3;
int Nx = Nx_per_unit*(Lx+3) + 1;
int Ny = Ny_per_unit*Ly + 1;
int Nx1 = (Nx-1)/(Lx+1)*9.0 + 1;
int Nx2 = (Nx-1)/(Lx+1)*(9.5 + 0.5*fine_para) + 1;
int Nx3 = (Nx-1)/(Lx+1)*(10.0 + fine_para) + 1;
double beta = 2.5;
double theta = 32.07;
double dx = Lx/(Nx - 1);
double dy = Ly/(Ny - 1);
double T = 30.00;
double dt = 0.002;
int Nt = int(T/dt + 0.5) + 1;
// fluid properties
double gamma = 1.4;
double fm0 = 2.0;
double angle = 20.0;
double u0 = 1.0;
double v0 = 0.0;
//double u0 = 1.0*cos(20.0*M_PI/180.0);
//double v0 = -1.0*sin(20.0*M_PI/180.0);
double p0 = 1.0/(gamma*fm0*fm0);
double rho0 = 1.0;
double t0 = 1.0;
double Rgas = 1.0/(gamma*fm0*fm0);
double Cp = Rgas*gamma/(gamma - 1.0);
double H_total = 0.5*(u0*u0 + v0*v0) + t0*Cp;
double del = 0.1;	// Entropy fix parameter
// grid parameters
double *Jac_c, *dxgdx_c, *dxgdy_c, *dygdx_c, *dygdy_c;
double *Jac_x, *dxgdx_x, *dxgdy_x, *dygdx_x, *dygdy_x;
double *Jac_y, *dxgdx_y, *dxgdy_y, *dygdx_y, *dygdy_y;
double Lxg = 1.0;
double Lyg = 1.0;
double dxg = Lxg/(Nx - 1);
double dyg = Lyg/(Ny - 1);
int Nx_mod = 10 + 1;
int Ny_mod = 10 + 1;
double r_mod = 0.8;
// Global variables
double *W1, *W2, *W3, *W4;	// Primitive var: 1. rho, 2. u, 3. v, 4. pressure
double *Tem;				// Temperature
double *Q1, *Q2, *Q3, *Q4;	// Conserved quantities
double *E1, *E2, *E3, *E4;	// Flux in x-dir
double *F1, *F2, *F3, *F4;	// Flux in y-dir
double *X, *Y;				// Spatial coordinate x
// function delaration
/// function of grid
void mem_alloc();
void mem_free();
void grid_gen();
void grid_bc();
void grid_modify();
void cal_tran_para_c();
void cal_tran_para_x();
void cal_tran_para_y();
void output_grid();
/// function of flow field
void init();
void cal_BC();
void cal_Flux_x();
void cal_Quan_x();
void cal_Flux_y();
void cal_Quan_y();
void cal_Prim();
double minmod(double a, double b, double c);
double minmod2(double a, double b, double c, double d);
void cal_W2Q(int i, int j);
void cal_WQ2E(int i, int j, double*E);
void cal_WQ2F(int i, int j, double*F);
void cal_Q2W(int i, int j);
void output();


int main() {
	clock_t t_start, t_end;
	
	mem_alloc();
	grid_gen();
	//grid_modify();
	cal_tran_para_c();
	cal_tran_para_x();
	cal_tran_para_y();
	output_grid();

	init();
	cal_BC();
	
	t_start = clock();
	cout << "Total = " << Nt << endl;
	for (int t = 0; t<Nt; t++) {
		// x-dir
		cal_Flux_x();
		cal_Quan_x();
		cal_Prim();
		cal_BC();
		// y-dir
		cal_Flux_y();
		cal_Quan_y();
		cal_Prim();
		cal_BC();
		t++;
		// y-dir
		cal_Flux_y();
		cal_Quan_y();
		cal_Prim();
		cal_BC();
		// x-dir
		cal_Flux_x();
		cal_Quan_x();
		cal_Prim();
		cal_BC();

		if (t%11==0)
			cout << "\rt = " << t;
	}
	cout << endl << "Calculation is complete !!" << endl;
	t_end = clock();
	cout << "Time (sec) = " << (t_end - t_start)/(double)(CLOCKS_PER_SEC) << endl;
	output();
	mem_free();


	system("pause");
	return 0;
}


void mem_alloc() {
	W1 = new double[Nx*Ny];
	W2 = new double[Nx*Ny];
	W3 = new double[Nx*Ny];
	W4 = new double[Nx*Ny];
	Q1 = new double[Nx*Ny];
	Q2 = new double[Nx*Ny];
	Q3 = new double[Nx*Ny];
	Q4 = new double[Nx*Ny];
	E1 = new double[(Nx-1)*Ny];
	E2 = new double[(Nx-1)*Ny];
	E3 = new double[(Nx-1)*Ny];
	E4 = new double[(Nx-1)*Ny];
	F1 = new double[Nx*(Ny-1)];
	F2 = new double[Nx*(Ny-1)];
	F3 = new double[Nx*(Ny-1)];
	F4 = new double[Nx*(Ny-1)];
	Tem = new double[Nx*Ny];
	X = new double[Nx*Ny];
	Y = new double[Nx*Ny];
	Jac_c = new double[Nx*Ny];
	dxgdx_c = new double[Nx*Ny];
	dxgdy_c = new double[Nx*Ny];
	dygdx_c = new double[Nx*Ny];
	dygdy_c = new double[Nx*Ny];
	Jac_x = new double[Nx*Ny];
	dxgdx_x = new double[Nx*Ny];
	dxgdy_x = new double[Nx*Ny];
	dygdx_x = new double[Nx*Ny];
	dygdy_x = new double[Nx*Ny];
	Jac_y = new double[Nx*Ny];
	dxgdx_y = new double[Nx*Ny];
	dxgdy_y = new double[Nx*Ny];
	dygdx_y = new double[Nx*Ny];
	dygdy_y = new double[Nx*Ny];
}


void mem_free() {
	delete W1;
	delete W2;
	delete W3;
	delete W4;
	delete Q1;
	delete Q2;
	delete Q3;
	delete Q4;
	delete E1;
	delete E2;
	delete E3;
	delete E4;
	delete F1;
	delete F2;
	delete F3;
	delete F4;
	delete Tem;
	delete X;
	delete Y;
	delete Jac_c;
	delete dxgdx_c;
	delete dxgdy_c;
	delete dygdx_c;
	delete dygdy_c;
	delete Jac_x;
	delete dxgdx_x;
	delete dxgdy_x;
	delete dygdx_x;
	delete dygdy_x;
	delete Jac_y;
	delete dxgdx_y;
	delete dxgdy_y;
	delete dygdx_y;
	delete dygdy_y;
}


void grid_gen() {
	double dx_in;
	double dy_in;
	double dx_t;
	// Calculate grid boundary
	grid_bc();

	for (int j=1; j<Ny-1; j++) {
		for (int i=1; i<Nx-1; i++) {
			Y[j*Nx+i] = Y[i] + (Y[(Ny-1)*Nx+i] - Y[i])*((Y[j*Nx] - Y[0])/(Y[(Ny-1)*Nx] - Y[0]));
		}
	}
	for (int i=1; i<Nx-1; i++) {
		for (int j=1; j<Ny-1; j++) {
			X[j*Nx+i] = X[j*Nx] + (X[j*Nx+(Nx-1)] - X[j*Nx])*((X[i] - X[0])/(X[(Nx-1)] - X[0]));
		}
	}
}


void grid_bc() {
	double dx = Lx/(Nx-1);
	double dy = Ly/(Ny-1);
	double dx_temp = 0.5/(Nx2 - Nx1);
	double dy_temp = 0.5*tan(beta/180*M_PI)/(Nx2 - Nx1);
	double rat, x_temp, x_temp2, y_temp, y_temp2;
	// Top
	/// sect 1
	dx_temp = 9.0/(Nx1 - 1);
	for (int i=0; i<Nx1; i++) {
		rat = (double)(Nx1-1-i)/(Nx1-1)*0.9 + 0.1;
		X[(Ny-1)*Nx+i] = 0.0 - (Nx1-1-i)*dx_temp*rat;
		Y[(Ny-1)*Nx+i] = 0.0;
	}
	/// sect 2
	x_temp = 0.5;
	y_temp = -0.5*tan(beta/180*M_PI);
	dx_temp = x_temp/(Nx2 - Nx1);
	dy_temp = y_temp/(Nx2 - Nx1);
	for (int i=Nx1; i<Nx2; i++) {
		if (i<((Nx1-1) + (Nx2-Nx1)/2)) {
			rat = (double)(i - (Nx1-1))/(0.5*(Nx2-Nx1))*0.9 + 0.1;
			X[(Ny-1)*Nx+i] = 0.0 + (i - (Nx1-1))*dx_temp*rat;
			Y[(Ny-1)*Nx+i] = 0.0 + (i - (Nx1-1))*dy_temp*rat;
		}
		else if (i==((Nx1-1) + (Nx2-Nx1)/2)) {
			rat = 1.0;
			X[(Ny-1)*Nx+i] = 0.0 + (i - (Nx1-1))*dx_temp*rat;
			Y[(Ny-1)*Nx+i] = 0.0 + (i - (Nx1-1))*dy_temp*rat;
		}
		else {
			rat = (double)((Nx2-1) - i)/(0.5*(Nx2-Nx1))*0.9 + 0.1;
			X[(Ny-1)*Nx+i] = x_temp - ((Nx2-1) - i)*dx_temp*rat;
			Y[(Ny-1)*Nx+i] = y_temp - ((Nx2-1) - i)*dy_temp*rat;
		}
	}
	/// sect 3
	x_temp = X[(Ny-1)*Nx+(Nx2-1)];
	y_temp = Y[(Ny-1)*Nx+(Nx2-1)];
	x_temp2 = 1.0;
	y_temp2 = 0.0;
	dx_temp = 0.5/(Nx3 - Nx2);
	dy_temp = 0.5*tan(beta/180*M_PI)/(Nx3 - Nx2);
	for (int i=Nx2; i<Nx3; i++) {
		if (i<((Nx2-1) + (Nx3-Nx2)/2)) {
			rat = (double)(i - (Nx2-1))/(0.5*(Nx3-Nx2))*0.9 + 0.1;
			X[(Ny-1)*Nx+i] = x_temp + (i - (Nx2-1))*dx_temp*rat;
			Y[(Ny-1)*Nx+i] = y_temp + (i - (Nx2-1))*dy_temp*rat;
		}
		else if (i==((Nx2-1) + (Nx3-Nx2)/2)) {
			rat = 1.0;
			X[(Ny-1)*Nx+i] = x_temp + (i - (Nx2-1))*dx_temp*rat;
			Y[(Ny-1)*Nx+i] = y_temp + (i - (Nx2-1))*dy_temp*rat;
		}
		else {
			rat = (double)((Nx3-1) - i)/(0.5*(Nx3-Nx2))*0.9 + 0.1;
			X[(Ny-1)*Nx+i] = x_temp2 - ((Nx3-1) - i)*dx_temp*rat;
			Y[(Ny-1)*Nx+i] = y_temp2 - ((Nx3-1) - i)*dy_temp*rat;
		}
	}
	/// sect 4
	dx_temp = 10.0/(Nx - Nx3);
	for (int i=Nx3; i<Nx; i++) {
		rat = (double)(i-(Nx3-1))/(Nx-Nx3)*0.9 + 0.1;
		X[(Ny-1)*Nx+i] = 1.0 + (i - (Nx3-1))*dx_temp*rat;
		Y[(Ny-1)*Nx+i] = 0.0;
	}
	// Bottom
	/// sect 1
	dx_temp = 9.0/(Nx1 - 1);
	x_temp = (Ly*tan((90.0-theta)/180*M_PI));
	for (int i=0; i<Nx1; i++) {
		rat = (double)(Nx1-1-i)/(Nx1-1)*0.9 + 0.1;
		X[i] = x_temp - (Nx1-1-i)*dx_temp*rat;
		Y[i] = -Ly;
	}
	/// sect 2
	x_temp = X[Nx1-1];
	x_temp2 = X[Nx1-1] + 0.5;
	dx_temp = 0.5/(Nx2 - Nx1);
	for (int i=Nx1; i<Nx2; i++) {
		if (i<((Nx1-1) + (Nx2-Nx1)/2)) {
			rat = (double)(i - (Nx1-1))/(0.5*(Nx2-Nx1))*0.9 + 0.1;
			X[i] = x_temp + (i - (Nx1-1))*dx_temp*rat;
			Y[i] = -Ly;
		}
		else if (i==((Nx1-1) + (Nx2-Nx1)/2)) {
			rat = 1.0;
			X[i] = x_temp + (i - (Nx1-1))*dx_temp*rat;
			Y[i] = -Ly;
		}
		else {
			rat = (double)((Nx2-1) - i)/(0.5*(Nx2-Nx1))*0.9 + 0.1;
			X[i] = x_temp2 - ((Nx2-1) - i)*dx_temp*rat;
			Y[i] = -Ly;
		}
	}
	/// sect 3
	x_temp = X[Nx2-1];
	x_temp2 = X[Nx2-1] + 0.5;
	dx_temp = 0.5/(Nx3 - Nx2);
	for (int i=Nx2; i<Nx3; i++) {
		if (i<((Nx2-1) + (Nx3-Nx2)/2)) {
			rat = (double)(i - (Nx2-1))/(0.5*(Nx3-Nx2))*0.9 + 0.1;
			X[i] = x_temp + (i - (Nx2-1))*dx_temp*rat;
			Y[i] = -Ly;
		}
		else if (i==((Nx2-1) + (Nx3-Nx2)/2)) {
			rat = 1.0;
			X[i] = x_temp + (i - (Nx2-1))*dx_temp*rat;
			Y[i] = -Ly;
		}
		else {
			rat = (double)((Nx3-1) - i)/(0.5*(Nx3-Nx2))*0.9 + 0.1;
			X[i] = x_temp2 - ((Nx3-1) - i)*dx_temp*rat;
			Y[i] = -Ly;
		}
	}
	/// sect 4
	dx_temp = 10.0/(Nx - Nx3);
	for (int i=Nx3; i<Nx; i++) {
		rat = (double)(i-(Nx3-1))/(Nx-Nx3)*0.9 + 0.1;
		X[i] = X[(Nx3-1)] + (i - (Nx3-1))*dx_temp*rat;
		Y[i] = -Ly;
	}
	// Left
	dx_temp = Ly*tan((90.0-theta)/180*M_PI)/(Ny - 1);
	dy_temp = -Ly/(Ny - 1);
	x_temp = X[0];
	y_temp = Y[0];
	for (int j=1; j<Ny-1; j++) {
		if (j>((Ny-1)/2+1-1)) {
			rat = (double)(Ny-1-j)/(0.5*(Ny-1))*0.9 + 0.1;
			X[j*Nx] = -9.0 + (Ny-1-j)*dx_temp*rat;
			Y[j*Nx] = (Ny-1-j)*dy_temp*rat;
		}
		else if (j==((Ny-1)/2+1-1)) {
			rat = 1.0;
			X[j*Nx] = -9.0 + (Ny-1-j)*dx_temp*rat;
			Y[j*Nx] = (Ny-1-j)*dy_temp*rat;
		}
		else {
			rat = (double)j/(0.5*(Ny-1))*0.9 + 0.1;
			X[j*Nx] = x_temp - j*dx_temp*rat;
			Y[j*Nx] = y_temp - j*dy_temp*rat;
		}
	}
	// Right
	dx_temp = Ly*tan((90.0-theta)/180*M_PI)/(Ny - 1);
	dy_temp = -Ly/(Ny - 1);
	x_temp = X[Nx-1];
	y_temp = Y[Nx-1];
	for (int j=1; j<Ny-1; j++) {
		if (j>((Ny-1)/2+1-1)) {
			rat = (double)(Ny-1-j)/(0.5*(Ny-1))*0.9 + 0.1;
			X[j*Nx+(Nx-1)] = 11.0 + (Ny-1-j)*dx_temp*rat;
			Y[j*Nx+(Nx-1)] = (Ny-1-j)*dy_temp*rat;
		}
		else if (j==((Ny-1)/2+1-1)) {
			rat = 1.0;
			X[j*Nx+(Nx-1)] = 11.0 + (Ny-1-j)*dx_temp*rat;
			Y[j*Nx+(Nx-1)] = (Ny-1-j)*dy_temp*rat;
		}
		else {
			rat = (double)j/(0.5*(Ny-1))*0.9 + 0.1;
			X[j*Nx+(Nx-1)] = x_temp - j*dx_temp*rat;
			Y[j*Nx+(Nx-1)] = y_temp - j*dy_temp*rat;
		}
	}
}


void grid_modify() {
	double tx, ty;
	double nx, ny;
	double s_para, w_para, m;
	double x_mod, y_mod;

	// Modify bottom row
	w_para = 1.0;
	for (int j=1; j<Ny_mod; j++) {
		for (int i=1; i<Nx-1; i++) {
			tx = X[(j-1)*Nx+(i+1)] - X[(j-1)*Nx+(i-1)];
			ty = Y[(j-1)*Nx+(i+1)] - Y[(j-1)*Nx+(i-1)];
			nx = ty;
			ny = -tx;
			// left line
			m = (Y[j*Nx+i] - Y[j*Nx+(i-1)])\
				/(X[j*Nx+i] - X[j*Nx+(i-1)]);
			s_para = (m*(X[(j-1)*Nx+i] - X[j*Nx+(i-1)]) \
					  - (Y[(j-1)*Nx+i] - Y[j*Nx+(i-1)]))/(ny - m*nx);
			x_mod = X[(j-1)*Nx+i] + s_para*nx;
			y_mod = Y[(j-1)*Nx+i] + s_para*ny;
			// w = 1 near bc, and w = 0 near center
			x_mod = X[j*Nx+i] + w_para*(x_mod - X[j*Nx+i]);
			y_mod = Y[j*Nx+i] + w_para*(y_mod - Y[j*Nx+i]);
			if (x_mod < X[j*Nx+(i-1)]) {
				cout << "non-modified in i=" << i << ", j=" << j << endl;
			}
			else if (x_mod < X[j*Nx+i]) {
				X[j*Nx+i] = x_mod;
				Y[j*Nx+i] = y_mod;
			}
			else {
				// right line
				m = (Y[j*Nx+(i+1)] - Y[j*Nx+i])\
					/(X[j*Nx+(i+1)] - X[j*Nx+i]);
				s_para = (m*(X[(j-1)*Nx+i] - X[j*Nx+i]) \
						  - (Y[(j-1)*Nx+i] - Y[j*Nx+i]))/(ny - m*nx);
				x_mod = X[(j-1)*Nx+i] + s_para*nx;
				y_mod = Y[(j-1)*Nx+i] + s_para*ny;
				// w = 1 near bc, and w = 0 near center
				x_mod = X[j*Nx+i] + w_para*(x_mod - X[j*Nx+i]);
				y_mod = Y[j*Nx+i] + w_para*(y_mod - Y[j*Nx+i]);
				if (x_mod > X[j*Nx+(i+1)]) {
					cout << "non-modified in i=" << i << ", j=" << j << endl;
				}
				else {
					X[j*Nx+i] = x_mod;
					Y[j*Nx+i] = y_mod;
				}
			}
		}
		w_para = r_mod*w_para;
	}
	// Modify top row
	w_para = 1.0;
	for (int j=Ny-2; j>((Ny-1)-Ny_mod); j--) {
		for (int i=1; i<Nx-1; i++) {
			tx = X[(j+1)*Nx+(i+1)] - X[(j+1)*Nx+(i-1)];
			ty = Y[(j+1)*Nx+(i+1)] - Y[(j+1)*Nx+(i-1)];
			nx = ty;
			ny = -tx;
			// left line
			m = (Y[j*Nx+i] - Y[j*Nx+(i-1)])\
				/(X[j*Nx+i] - X[j*Nx+(i-1)]);
			s_para = (m*(X[(j+1)*Nx+i] - X[j*Nx+(i-1)]) \
					  - (Y[(j+1)*Nx+i] - Y[j*Nx+(i+1)]))/(ny - m*nx);
			x_mod = X[(j+1)*Nx+i] + s_para*nx;
			y_mod = Y[(j+1)*Nx+i] + s_para*ny;
			// w = 1 near bc, and w = 0 near center
			x_mod = X[j*Nx+i] + w_para*(x_mod - X[j*Nx+i]);
			y_mod = Y[j*Nx+i] + w_para*(y_mod - Y[j*Nx+i]);
			if (x_mod < X[j*Nx+(i-1)]) {
				cout << "non-modified in i=" << i << ", j=" << j << endl;
			}
			else if (x_mod < X[j*Nx+i]) {
				X[j*Nx+i] = x_mod;
				Y[j*Nx+i] = y_mod;
			}
			else {
				// right line
				m = (Y[j*Nx+(i+1)] - Y[j*Nx+i])\
					/(X[j*Nx+(i+1)] - X[j*Nx+i]);
				s_para = (m*(X[(j+1)*Nx+i] - X[j*Nx+i]) \
						  - (Y[(j+1)*Nx+i] - Y[j*Nx+i]))/(ny - m*nx);
				x_mod = X[(j+1)*Nx+i] + s_para*nx;
				y_mod = Y[(j+1)*Nx+i] + s_para*ny;
				// w = 1 near bc, and w = 0 near center
				x_mod = X[j*Nx+i] + w_para*(x_mod - X[j*Nx+i]);
				y_mod = Y[j*Nx+i] + w_para*(y_mod - Y[j*Nx+i]);
				if (x_mod > X[j*Nx+(i+1)]) {
					cout << "non-modified in i=" << i << ", j=" << j << endl;
				}
				else {
					X[j*Nx+i] = x_mod;
					Y[j*Nx+i] = y_mod;
				}
			}
		}
		w_para = r_mod*w_para;
	}
	//	 Modify left column
	w_para = 1.0;
	for (int i=1; i<Nx_mod; i++) {
		for (int j=1; j<Ny-1; j++) {
			tx = X[(j+1)*Nx+(i-1)] - X[(j-1)*Nx+(i-1)];
			ty = Y[(j+1)*Nx+(i-1)] - Y[(j-1)*Nx+(i-1)];
			nx = ty;
			ny = -tx;
			// down line
			m = (X[(j-1)*Nx+i] - X[j*Nx+i])\
				/(Y[(j-1)*Nx+i] - Y[j*Nx+i]);
			s_para = ((X[j*Nx+(i-1)] - X[(j-1)*Nx+i]) \
					  - m*(Y[j*Nx+(i-1)] - Y[(j-1)*Nx+i]))/(m*ny - nx);
			x_mod = X[j*Nx+(i-1)] + s_para*nx;
			y_mod = Y[j*Nx+(i-1)] + s_para*ny;
			// w = 1 near bc, and w = 0 near center
			x_mod = X[j*Nx+i] + w_para*(x_mod - X[j*Nx+i]);
			y_mod = Y[j*Nx+i] + w_para*(y_mod - Y[j*Nx+i]);
			if (y_mod < Y[(j-1)*Nx+i]) {
				cout << "non-modified in i=" << i << ", j=" << j << endl;
			}
			else if (y_mod < Y[j*Nx+i]) {
				X[j*Nx+i] = x_mod;
				Y[j*Nx+i] = y_mod;
			}
			else {
				// up line
				m = (X[(j+1)*Nx+i] - X[j*Nx+i])\
					/(Y[(j+1)*Nx+i] - Y[j*Nx+i]);
				s_para = ((X[j*Nx+(i-1)] - X[j*Nx+i]) \
						  - m*(Y[j*Nx+(i-1)] - Y[j*Nx+i]))/(m*ny - nx);
				x_mod = X[j*Nx+(i-1)] + s_para*nx;
				y_mod = Y[j*Nx+(i-1)] + s_para*ny;
				// w = 1 near bc, and w = 0 near center
				x_mod = X[j*Nx+i] + w_para*(x_mod - X[j*Nx+i]);
				y_mod = Y[j*Nx+i] + w_para*(y_mod - Y[j*Nx+i]);
				if (y_mod > Y[(j+1)*Nx+i]) {
					cout << "non-modified in i=" << i << ", j=" << j << endl;
				}
				else {
					X[j*Nx+i] = x_mod;
					Y[j*Nx+i] = y_mod;
				}
			}
		}
		w_para = r_mod*w_para;
	}
	//	 Modify right column
	w_para = 1.0;
	for (int i=Nx-2; i>((Nx-1)-Nx_mod); i--) {
		for (int j=1; j<Ny-1; j++) {
			tx = X[(j+1)*Nx+(i+1)] - X[(j-1)*Nx+(i+1)];
			ty = Y[(j+1)*Nx+(i+1)] - Y[(j-1)*Nx+(i+1)];
			nx = ty;
			ny = -tx;
			// down line
			m = (X[(j-1)*Nx+i] - X[j*Nx+i])\
				/(Y[(j-1)*Nx+i] - Y[j*Nx+i]);
			s_para = ((X[j*Nx+(i+1)] - X[(j-1)*Nx+i]) \
					  - m*(Y[j*Nx+(i+1)] - Y[(j-1)*Nx+i]))/(m*ny - nx);
			x_mod = X[j*Nx+(i+1)] + s_para*nx;
			y_mod = Y[j*Nx+(i+1)] + s_para*ny;
			// w = 1 near bc, and w = 0 near center
			x_mod = X[j*Nx+i] + w_para*(x_mod - X[j*Nx+i]);
			y_mod = Y[j*Nx+i] + w_para*(y_mod - Y[j*Nx+i]);
			if (y_mod < Y[(j-1)*Nx+i]) {
				cout << "non-modified in i=" << i << ", j=" << j << endl;
			}
			else if (y_mod < Y[j*Nx+i]) {
				X[j*Nx+i] = x_mod;
				Y[j*Nx+i] = y_mod;
			}
			else {
				// up line
				m = (X[(j+1)*Nx+i] - X[j*Nx+i])\
					/(Y[(j+1)*Nx+i] - Y[j*Nx+i]);
				s_para = ((X[j*Nx+(i+1)] - X[(j+1)*Nx+i]) \
						  - m*(Y[j*Nx+(i+1)] - Y[(j+1)*Nx+i]))/(m*ny - nx);
				x_mod = X[j*Nx+(i+1)] + s_para*nx;
				y_mod = Y[j*Nx+(i+1)] + s_para*ny;
				// w = 1 near bc, and w = 0 near center
				x_mod = X[j*Nx+i] + w_para*(x_mod - X[j*Nx+i]);
				y_mod = Y[j*Nx+i] + w_para*(y_mod - Y[j*Nx+i]);
				if (y_mod > Y[(j+1)*Nx+i]) {
					cout << "non-modified in i=" << i << ", j=" << j << endl;
				}
				else {
					X[j*Nx+i] = x_mod;
					Y[j*Nx+i] = y_mod;
				}
			}
		}
		w_para = r_mod*w_para;
	}
}


void cal_tran_para_c() {
	double dxdxg, dxdyg, dydxg, dydyg;

	for (int j=1; j<Ny-1; j++)
		for (int i=1; i<Nx-1; i++) {
			dxdxg = (X[j*Nx+(i+1)] - X[j*Nx+(i-1)])/(2.0*dxg);
			dydxg = (Y[j*Nx+(i+1)] - Y[j*Nx+(i-1)])/(2.0*dxg);
			dxdyg = (X[(j+1)*Nx+i] - X[(j-1)*Nx+i])/(2.0*dyg);
			dydyg = (Y[(j+1)*Nx+i] - Y[(j-1)*Nx+i])/(2.0*dyg);
			Jac_c[j*Nx+i] = 1.0/(dxdxg*dydyg - dxdyg*dydxg);
			dxgdx_c[j*Nx+i] = Jac_c[j*Nx+i]*dydyg;
			dxgdy_c[j*Nx+i] = -Jac_c[j*Nx+i]*dxdyg;
			dygdx_c[j*Nx+i] = -Jac_c[j*Nx+i]*dydxg;
			dygdy_c[j*Nx+i] = Jac_c[j*Nx+i]*dxdxg;
		}
	//// bc
	//// left-bottom
	//dxdxg = (-3.0*X[0*Nx+0] + 4.0*X[0*Nx+(0+1)] - X[0*Nx+(0+2)])/(2.0*dxg);
	//dydxg = (-3.0*Y[0*Nx+0] + 4.0*Y[0*Nx+(0+1)] - Y[0*Nx+(0+2)])/(2.0*dxg);
	//dxdyg = (-3.0*X[0*Nx+0] + 4.0*X[(0+1)*Nx+0] - X[(0+2)*Nx+0])/(2.0*dyg);
	//dydyg = (-3.0*Y[0*Nx+0] + 4.0*Y[(0+1)*Nx+0] - Y[(0+2)*Nx+0])/(2.0*dyg);
	//Jac_c[0*Nx+0] = 1.0/(dxdxg*dydyg - dxdyg*dydxg);
	//dxgdx_c[0*Nx+0] = Jac_c[0*Nx+0]*dydyg;
	//dxgdy_c[0*Nx+0] = -Jac_c[0*Nx+0]*dxdyg;
	//dygdx_c[0*Nx+0] = -Jac_c[0*Nx+0]*dydxg;
	//dygdy_c[0*Nx+0] = Jac_c[0*Nx+0]*dxdxg;
	//// right-bottom
	//dxdxg = (3.0*X[0*Nx+(Nx-1)] - 4.0*X[0*Nx+(Nx-2)] + X[0*Nx+(Nx-3)])/(2.0*dxg);
	//dydxg = (3.0*Y[0*Nx+(Nx-1)] - 4.0*Y[0*Nx+(Nx-2)] + Y[0*Nx+(Nx-3)])/(2.0*dxg);
	//dxdyg = (-3.0*X[0*Nx+(Nx-1)] + 4.0*X[(0+1)*Nx+(Nx-1)] - X[(0+2)*Nx+(Nx-1)])/(2.0*dyg);
	//dydyg = (-3.0*Y[0*Nx+(Nx-1)] + 4.0*Y[(0+1)*Nx+(Nx-1)] - Y[(0+2)*Nx+(Nx-1)])/(2.0*dyg);
	//Jac_c[0*Nx+0] = 1.0/(dxdxg*dydyg - dxdyg*dydxg);
	//dxgdx_c[0*Nx+(Nx-1)] = Jac_c[0*Nx+(Nx-1)]*dydyg;
	//dxgdy_c[0*Nx+(Nx-1)] = -Jac_c[0*Nx+(Nx-1)]*dxdyg;
	//dygdx_c[0*Nx+(Nx-1)] = -Jac_c[0*Nx+(Nx-1)]*dydxg;
	//dygdy_c[0*Nx+(Nx-1)] = Jac_c[0*Nx+(Nx-1)]*dxdxg;
	//// left-top
	//dxdxg = (-3.0*X[(Ny-1)*Nx+0] + 4.0*X[(Ny-1)*Nx+(0+1)] - X[(Ny-1)*Nx+(0+2)])/(2.0*dxg);
	//dydxg = (-3.0*Y[(Ny-1)*Nx+0] + 4.0*Y[(Ny-1)*Nx+(0+1)] - Y[(Ny-1)*Nx+(0+2)])/(2.0*dxg);
	//dxdyg = (3.0*X[(Ny-1)*Nx+0] - 4.0*X[(Ny-2)*Nx+0] + X[(Ny-3)*Nx+0])/(2.0*dyg);
	//dydyg = (3.0*Y[(Ny-1)*Nx+0] - 4.0*Y[(Ny-2)*Nx+0] + Y[(Ny-3)*Nx+0])/(2.0*dyg);
	//Jac_c[(Ny-1)*Nx+0] = 1.0/(dxdxg*dydyg - dxdyg*dydxg);
	//dxgdx_c[(Ny-1)*Nx+0] = Jac_c[(Ny-1)*Nx+0]*dydyg;
	//dxgdy_c[(Ny-1)*Nx+0] = -Jac_c[(Ny-1)*Nx+0]*dxdyg;
	//dygdx_c[(Ny-1)*Nx+0] = -Jac_c[(Ny-1)*Nx+0]*dydxg;
	//dygdy_c[(Ny-1)*Nx+0] = Jac_c[(Ny-1)*Nx+0]*dxdxg;
	//// right-top
	//dxdxg = (3.0*X[(Ny-1)*Nx+(Nx-1)] - 4.0*X[(Ny-1)*Nx+(Nx-2)] + X[(Ny-1)*Nx+(Nx-3)])/(2.0*dxg);
	//dydxg = (3.0*Y[(Ny-1)*Nx+(Nx-1)] - 4.0*Y[(Ny-1)*Nx+(Nx-2)] + Y[(Ny-1)*Nx+(Nx-3)])/(2.0*dxg);
	//dxdyg = (3.0*X[(Ny-1)*Nx+(Nx-1)] - 4.0*X[(Ny-2)*Nx+(Nx-1)] + X[(Ny-3)*Nx+(Nx-1)])/(2.0*dyg);
	//dydyg = (3.0*Y[(Ny-1)*Nx+(Nx-1)] - 4.0*Y[(Ny-2)*Nx+(Nx-1)] + Y[(Ny-3)*Nx+(Nx-1)])/(2.0*dyg);
	//Jac_c[(Ny-1)*Nx+(Nx-1)] = 1.0/(dxdxg*dydyg - dxdyg*dydxg);
	//dxgdx_c[(Ny-1)*Nx+(Nx-1)] = Jac_c[(Ny-1)*Nx+(Nx-1)]*dydyg;
	//dxgdy_c[(Ny-1)*Nx+(Nx-1)] = -Jac_c[(Ny-1)*Nx+(Nx-1)]*dxdyg;
	//dygdx_c[(Ny-1)*Nx+(Nx-1)] = -Jac_c[(Ny-1)*Nx+(Nx-1)]*dydxg;
	//dygdy_c[(Ny-1)*Nx+(Nx-1)] = Jac_c[(Ny-1)*Nx+(Nx-1)]*dxdxg;
	//// left
	//for (int j=1; j<Ny-1; j++) {
	//	dxdxg = (-3.0*X[j*Nx+0] + 4.0*X[j*Nx+(0+1)] - X[j*Nx+(0+2)])/(2.0*dxg);
	//	dydxg = (-3.0*Y[j*Nx+0] + 4.0*Y[j*Nx+(0+1)] - Y[j*Nx+(0+2)])/(2.0*dxg);
	//	dxdyg = (X[(j+1)*Nx+0] - X[(j-1)*Nx+0])/(2.0*dyg);
	//	dydyg = (Y[(j+1)*Nx+0] - Y[(j-1)*Nx+0])/(2.0*dyg);
	//	Jac_c[j*Nx+0] = 1.0/(dxdxg*dydyg - dxdyg*dydxg);
	//	dxgdx_c[j*Nx+0] = Jac_c[j*Nx+0]*dydyg;
	//	dxgdy_c[j*Nx+0] = -Jac_c[j*Nx+0]*dxdyg;
	//	dygdx_c[j*Nx+0] = -Jac_c[j*Nx+0]*dydxg;
	//	dygdy_c[j*Nx+0] = Jac_c[j*Nx+0]*dxdxg;
	//}
	//// right
	//for (int j=1; j<Ny-1; j++) {
	//	dxdxg = (3.0*X[j*Nx+(Nx-1)] - 4.0*X[j*Nx+(Nx-2)] + X[j*Nx+(Nx-3)])/(2.0*dxg);
	//	dydxg = (3.0*Y[j*Nx+(Nx-1)] - 4.0*Y[j*Nx+(Nx-2)] + Y[j*Nx+(Nx-3)])/(2.0*dxg);
	//	dxdyg = (X[(j+1)*Nx+(Nx-1)] - X[(j-1)*Nx+(Nx-1)])/(2.0*dyg);
	//	dydyg = (Y[(j+1)*Nx+(Nx-1)] - Y[(j-1)*Nx+(Nx-1)])/(2.0*dyg);
	//	Jac_c[j*Nx+(Nx-1)] = 1.0/(dxdxg*dydyg - dxdyg*dydxg);
	//	dxgdx_c[j*Nx+(Nx-1)] = Jac_c[j*Nx+(Nx-1)]*dydyg;
	//	dxgdy_c[j*Nx+(Nx-1)] = -Jac_c[j*Nx+(Nx-1)]*dxdyg;
	//	dygdx_c[j*Nx+(Nx-1)] = -Jac_c[j*Nx+(Nx-1)]*dydxg;
	//	dygdy_c[j*Nx+(Nx-1)] = Jac_c[j*Nx+(Nx-1)]*dxdxg;
	//}
	//// bottom
	//for (int i=1; i<Nx-1; i++) {
	//	dxdxg = (X[0*Nx+(i+1)] - X[0*Nx+(i-1)])/(2.0*dxg);
	//	dydxg = (Y[0*Nx+(i+1)] - Y[0*Nx+(i-1)])/(2.0*dxg);
	//	dxdyg = (-3.0*X[0*Nx+i] + 4.0*X[(0+1)*Nx+i] - X[(0+2)*Nx+i])/(2.0*dyg);
	//	dydyg = (-3.0*Y[0*Nx+i] + 4.0*Y[(0+1)*Nx+i] - Y[(0+2)*Nx+i])/(2.0*dyg);
	//	Jac_c[0*Nx+i] = 1.0/(dxdxg*dydyg - dxdyg*dydxg);
	//	dxgdx_c[0*Nx+i] = Jac_c[0*Nx+i]*dydyg;
	//	dxgdy_c[0*Nx+i] = -Jac_c[0*Nx+i]*dxdyg;
	//	dygdx_c[0*Nx+i] = -Jac_c[0*Nx+i]*dydxg;
	//	dygdy_c[0*Nx+i] = Jac_c[0*Nx+i]*dxdxg;
	//}
	//// Top
	//for (int i=1; i<Nx-1; i++) {
	//	dxdxg = (X[(Ny-1)*Nx+(i+1)] - X[(Ny-1)*Nx+(i-1)])/(2.0*dxg);
	//	dydxg = (Y[(Ny-1)*Nx+(i+1)] - Y[(Ny-1)*Nx+(i-1)])/(2.0*dxg);
	//	dxdyg = (3.0*X[(Ny-1)*Nx+i] - 4.0*X[(Ny-2)*Nx+i] + X[(Ny-3)*Nx+i])/(2.0*dyg);
	//	dydyg = (3.0*Y[(Ny-1)*Nx+i] - 4.0*Y[(Ny-2)*Nx+i] + Y[(Ny-3)*Nx+i])/(2.0*dyg);
	//	Jac_c[(Ny-1)*Nx+i] = 1.0/(dxdxg*dydyg - dxdyg*dydxg);
	//	dxgdx_c[(Ny-1)*Nx+i] = Jac_c[(Ny-1)*Nx+i]*dydyg;
	//	dxgdy_c[(Ny-1)*Nx+i] = -Jac_c[(Ny-1)*Nx+i]*dxdyg;
	//	dygdx_c[(Ny-1)*Nx+i] = -Jac_c[(Ny-1)*Nx+i]*dydxg;
	//	dygdy_c[(Ny-1)*Nx+i] = Jac_c[(Ny-1)*Nx+i]*dxdxg;
	//}
}


void cal_tran_para_x() {
	double dxdxg, dxdyg, dydxg, dydyg;

	for (int j=1; j<Ny-1; j++)
		for (int i=0; i<Nx-1; i++) {
			dxdxg = (X[j*Nx+(i+1)] - X[j*Nx+i])/(dxg);
			dydxg = (Y[j*Nx+(i+1)] - Y[j*Nx+i])/(dxg);
			dxdyg = (0.5*(X[(j+1)*Nx+i] + X[(j+1)*Nx+(i+1)]) \
					 - 0.5*(X[(j-1)*Nx+i] + X[(j-1)*Nx+(i+1)]))/(2.0*dyg);
			dydyg = (0.5*(Y[(j+1)*Nx+i] + Y[(j+1)*Nx+(i+1)]) \
					 - 0.5*(Y[(j-1)*Nx+i] + Y[(j-1)*Nx+(i+1)]))/(2.0*dyg);
			Jac_x[j*Nx+i] = 1.0/(dxdxg*dydyg - dxdyg*dydxg);
			dxgdx_x[j*Nx+i] = Jac_x[j*Nx+i]*dydyg;
			dxgdy_x[j*Nx+i] = -Jac_x[j*Nx+i]*dxdyg;
			dygdx_x[j*Nx+i] = -Jac_x[j*Nx+i]*dydxg;
			dygdy_x[j*Nx+i] = Jac_x[j*Nx+i]*dxdxg;
		}
}


void cal_tran_para_y() {
	double dxdxg, dxdyg, dydxg, dydyg;

	for (int i=1; i<Nx-1; i++)
		for (int j=0; j<Ny-1; j++) {
			dxdxg = (0.5*(X[j*Nx+(i+1)] + X[(j+1)*Nx+(i+1)]) \
					 - 0.5*(X[j*Nx+(i-1)] + X[(j+1)*Nx+(i-1)]))/(2.0*dxg);
			dydxg = (0.5*(Y[j*Nx+(i+1)] + Y[(j+1)*Nx+(i+1)]) \
					 - 0.5*(Y[j*Nx+(i-1)] + Y[(j+1)*Nx+(i-1)]))/(2.0*dxg);
			dxdyg = (X[(j+1)*Nx+i] - X[j*Nx+i])/(dyg);
			dydyg = (Y[(j+1)*Nx+i] - Y[j*Nx+i])/(dyg);
			Jac_y[j*Nx+i] = 1.0/(dxdxg*dydyg - dxdyg*dydxg);
			dxgdx_y[j*Nx+i] = Jac_y[j*Nx+i]*dydyg;
			dxgdy_y[j*Nx+i] = -Jac_y[j*Nx+i]*dxdyg;
			dygdx_y[j*Nx+i] = -Jac_y[j*Nx+i]*dydxg;
			dygdy_y[j*Nx+i] = Jac_y[j*Nx+i]*dxdxg;
		}
}


void init() {
	for (int j = 0; j<Ny; j++) {
		for (int i = 0; i<Nx; i++) {
			W1[j*Nx+i] = rho0;
			W2[j*Nx+i] = u0;
			W3[j*Nx+i] = v0;
			W4[j*Nx+i] = p0;
			Tem[j*Nx+i] = t0;
			cal_W2Q(i, j);
		}
	}
}


void cal_BC() {
	double tx, ty, Vt, Vt1, Vt2;
	// Left
	for (int j = 0; j<Ny; j++) {
		W1[j*Nx+0] = rho0;
		W2[j*Nx+0] = u0;
		W3[j*Nx+0] = v0;
		W4[j*Nx+0] = p0;
		Tem[j*Nx+0] = t0;
		cal_W2Q(0, j);
	}
	// Top
	/// sect 1
	for (int i=0; i<Nx1-1; i++) {
		W2[(Ny-1)*Nx+i] = (4.0*W2[(Ny-2)*Nx+i] - W2[(Ny-3)*Nx+i])/3.0;
		W3[(Ny-1)*Nx+i] = 0.0;
		W4[(Ny-1)*Nx+i] = (4.0*W4[(Ny-2)*Nx+i] - W4[(Ny-3)*Nx+i])/3.0;
		Tem[(Ny-1)*Nx+i] = (H_total - 0.5*(W2[(Ny-1)*Nx+i]*W2[(Ny-1)*Nx+i] + W3[(Ny-1)*Nx+i]*W3[(Ny-1)*Nx+i]))/Cp;
		W1[(Ny-1)*Nx+i] = W4[(Ny-1)*Nx+i]/(Rgas*Tem[(Ny-1)*Nx+i]);
		cal_W2Q(i, Ny-1);
	}
	/// sect 2
	tx = cos(beta/180.0*M_PI);
	ty = -sin(beta/180.0*M_PI);
	for (int i=Nx1-1; i<Nx2-1; i++) {
		Vt1 = W2[(Ny-2)*Nx+i]*tx + W3[(Ny-2)*Nx+i]*ty;	// |Vt1| = u*tx + v*ty
		Vt2 = W2[(Ny-3)*Nx+i]*tx + W3[(Ny-3)*Nx+i]*ty;	// |Vt2| = u*tx + v*ty
		Vt = 2.0*Vt1 - Vt2;
		W2[(Ny-1)*Nx+i] = Vt*cos(beta/180.0*M_PI);
		W3[(Ny-1)*Nx+i] = -Vt*sin(beta/180.0*M_PI);
		W4[(Ny-1)*Nx+i] = (4.0*W4[(Ny-2)*Nx+i] - W4[(Ny-3)*Nx+i])/3.0;
		Tem[(Ny-1)*Nx+i] = (H_total - 0.5*(W2[(Ny-1)*Nx+i]*W2[(Ny-1)*Nx+i] + W3[(Ny-1)*Nx+i]*W3[(Ny-1)*Nx+i]))/Cp;
		W1[(Ny-1)*Nx+i] = W4[(Ny-1)*Nx+i]/(Rgas*Tem[(Ny-1)*Nx+i]);
		cal_W2Q(i, Ny-1);
	}
	/// Tip point
	W2[(Ny-1)*Nx+(Nx2-1)] = 2.0*W2[(Ny-2)*Nx+(Nx2-1)] - W2[(Ny-3)*Nx+(Nx2-1)];
	W3[(Ny-1)*Nx+(Nx2-1)] = 0.0;
	W4[(Ny-1)*Nx+(Nx2-1)] = (4.0*W4[(Ny-2)*Nx+(Nx2-1)] - W4[(Ny-3)*Nx+(Nx2-1)])/3.0;
	Tem[(Ny-1)*Nx+(Nx2-1)] = (H_total - 0.5*(W2[(Ny-1)*Nx+(Nx2-1)]*W2[(Ny-1)*Nx+(Nx2-1)] + W3[(Ny-1)*Nx+(Nx2-1)]*W3[(Ny-1)*Nx+(Nx2-1)]))/Cp;
	W1[(Ny-1)*Nx+(Nx2-1)] = W4[(Ny-1)*Nx+(Nx2-1)]/(Rgas*Tem[(Ny-1)*Nx+(Nx2-1)]);
	cal_W2Q((Nx2-1), Ny-1);
	/// sect 3
	tx = cos(beta/180*M_PI);
	ty = sin(beta/180*M_PI);
	for (int i=Nx2; i<Nx3; i++) {
		Vt1 = W2[(Ny-2)*Nx+i]*tx + W3[(Ny-2)*Nx+i]*ty;	// |Vt1| = u*tx + v*ty
		Vt2 = W2[(Ny-3)*Nx+i]*tx + W3[(Ny-3)*Nx+i]*ty;	// |Vt2| = u*tx + v*ty
		Vt = 2.0*Vt1 - Vt2;
		W2[(Ny-1)*Nx+i] = Vt*cos(beta/180*M_PI);
		W3[(Ny-1)*Nx+i] = Vt*sin(beta/180*M_PI);
		W4[(Ny-1)*Nx+i] = (4.0*W4[(Ny-2)*Nx+i] - W4[(Ny-3)*Nx+i])/3.0;
		Tem[(Ny-1)*Nx+i] = (H_total - 0.5*(W2[(Ny-1)*Nx+i]*W2[(Ny-1)*Nx+i] + W3[(Ny-1)*Nx+i]*W3[(Ny-1)*Nx+i]))/Cp;
		W1[(Ny-1)*Nx+i] = W4[(Ny-1)*Nx+i]/(Rgas*Tem[(Ny-1)*Nx+i]);
		cal_W2Q(i, Ny-1);
	}
	/// sect 4
	for (int i=Nx3; i<Nx; i++) {
		W2[(Ny-1)*Nx+i] = (4.0*W2[(Ny-2)*Nx+i] - W2[(Ny-3)*Nx+i])/3.0;
	//	W2[(Ny-1)*Nx+i] = 2.0*W2[(Ny-2)*Nx+i] - W2[(Ny-3)*Nx+i];
		W3[(Ny-1)*Nx+i] = 0.0;
		W4[(Ny-1)*Nx+i] = (4.0*W4[(Ny-2)*Nx+i] - W4[(Ny-3)*Nx+i])/3.0;
		Tem[(Ny-1)*Nx+i] = (H_total - 0.5*(W2[(Ny-1)*Nx+i]*W2[(Ny-1)*Nx+i] + W3[(Ny-1)*Nx+i]*W3[(Ny-1)*Nx+i]))/Cp;
		W1[(Ny-1)*Nx+i] = W4[(Ny-1)*Nx+i]/(Rgas*Tem[(Ny-1)*Nx+i]);
		cal_W2Q(i, Ny-1);
	}
	// Bottom
	tx = cos(0.0/180.0*M_PI);
	ty = sin(0.0/180.0*M_PI);
	for (int i = 0; i<Nx; i++) {
		Vt1 = W2[1*Nx+i]*tx + W3[1*Nx+i]*ty;	// |Vt1| = u*tx + v*ty
		Vt2 = W2[2*Nx+i]*tx + W3[2*Nx+i]*ty;	// |Vt2| = u*tx + v*ty
		Vt = 2.0*Vt1 - Vt2;
		W2[i] = Vt*cos(0.0/180*M_PI);
		W3[i] = Vt*sin(0.0/180*M_PI);
	//	W2[i] = 2.0*W2[1*Nx+i] - W2[2*Nx+i];
	//	W3[i] = 0.0;
		//W4[i] = W4[1*Nx+i];
		W4[i] = (4.0*W4[1*Nx+i] - W4[2*Nx+i])/3.0;
		Tem[i] = (H_total - 0.5*(W2[i]*W2[i] + W3[i]*W3[i]))/Cp;
		W1[i] = W4[i]/(Rgas*Tem[i]);
		cal_W2Q(i, 0);
	}
	// Right
	for (int j=0; j<Ny; j++) {
		W2[j*Nx+(Nx-1)] = W2[j*Nx+(Nx-2)];
		W3[j*Nx+(Nx-1)] = W3[j*Nx+(Nx-2)];
		W4[j*Nx+(Nx-1)] = W4[j*Nx+(Nx-2)];
		Tem[j*Nx+(Nx-1)] = (H_total - 0.5*(W2[j*Nx+(Nx-1)]*W2[j*Nx+(Nx-1)] \
							+ W3[j*Nx+(Nx-1)]*W3[j*Nx+(Nx-1)]))/Cp;
		W1[j*Nx+(Nx-1)] = W4[j*Nx+(Nx-1)]/(Rgas*Tem[j*Nx+(Nx-1)]);
		cal_W2Q(Nx-1, j);
	}
}


void cal_Flux_x() {
	double temp, temp2, sum;	// for sqrt test, and sum
	double D, *u_bar, *v_bar, *a_bar, *H_bar, H_L, H_R;
	double Ug, Vg;
	double b1, b2;
	double Rx[4][4], Rx_inv[4][4];
	double lam[4];
	double **alp, Q_del[4], alp_L[4], alp_R[4];	// quantity jump
	double q_limit[4];	// Flux limiter
	double phi[4], psi;
	double E_L[4], E_R[4], F_L[4], F_R[4];	// Flux on the left and right hand sides
	double del_x, lam_ref;
	double *k1, *k2, *xgyg_sq;

	// Memory Allocation
	u_bar = new double[Nx-1];
	v_bar = new double[Nx-1];
	a_bar = new double[Nx-1];
	H_bar = new double[Nx-1];
	k1 = new double[Nx-1];
	k2 = new double[Nx-1];
	xgyg_sq = new double[Nx-1];
	alp = new double*[4];
	for (int k = 0; k<4; k++) {
		alp[k] = new double[Nx-1];
	}

	for (int j = 1; j<Ny-1; j++) {
		// Calculate quantity jump
		for (int i = 0; i<Nx-1; i++) {
			// Roe average
			/// sqrt for D
			temp = W1[j*Nx+(i+1)]/W1[j*Nx+i];
			if (temp>=0.0)
				D = sqrt(temp);
			else {
				cout<<"domain error in D, i = "<<endl;
				output();
				system("pause");
				exit(-1);
			}

			u_bar[i] = (W2[j*Nx+i] + D*W2[j*Nx+(i+1)])/(1.0 + D);
			v_bar[i] = (W3[j*Nx+i] + D*W3[j*Nx+(i+1)])/(1.0 + D);
			H_L = (Q4[j*Nx+i] + W4[j*Nx+i])/W1[j*Nx+i];
			H_R = (Q4[j*Nx+(i+1)] + W4[j*Nx+(i+1)])/W1[j*Nx+(i+1)];
			H_bar[i] = (H_L + D*H_R)/(1.0 + D);
			/// Roe average a
			temp = (gamma - 1.0)*(H_bar[i] - 0.5*(u_bar[i]*u_bar[i] + v_bar[i]*v_bar[i]));
			//temp2 = min(gamma*W4[j*Nx+i]/W1[j*Nx+i], gamma*W4[j*Nx+(i+1)]/W1[j*Nx+(i+1)]);
			//temp = max(temp, temp2);
			if (temp>=0.0)
				a_bar[i] = sqrt(temp);
			else {
				cout<<"domain error in a_bar, i = "<<endl;
				output();
				system("pause");
				exit(-1);
			}
			/// b1, b2, k1, k2
			b2 = (gamma - 1.0)/(a_bar[i]*a_bar[i]);
			b1 = 0.5*b2*(u_bar[i]*u_bar[i] + v_bar[i]*v_bar[i]);
			xgyg_sq[i] = sqrt(dxgdx_x[j*Nx+i]*dxgdx_x[j*Nx+i] + dxgdy_x[j*Nx+i]*dxgdy_x[j*Nx+i]);
			k1[i] = dxgdx_x[j*Nx+i]/xgyg_sq[i];
			k2[i] = dxgdy_x[j*Nx+i]/xgyg_sq[i];
			// Inverse of eigenvector
			Rx_inv[0][0] = 0.5*(b1 + k1[i]*u_bar[i]/a_bar[i] + k2[i]*v_bar[i]/a_bar[i]);
			Rx_inv[0][1] = -0.5*(b2*u_bar[i] + k1[i]/a_bar[i]);
			Rx_inv[0][2] = -0.5*(b2*v_bar[i] + k2[i]/a_bar[i]);
			Rx_inv[0][3] = 0.5*b2;
			Rx_inv[1][0] = 1.0-b1;
			Rx_inv[1][1] = b2*u_bar[i];
			Rx_inv[1][2] = b2*v_bar[i];
			Rx_inv[1][3] = -b2;
			Rx_inv[2][0] = 0.5*(b1 - k1[i]*u_bar[i]/a_bar[i] - k2[i]*v_bar[i]/a_bar[i]);
			Rx_inv[2][1] = -0.5*(b2*u_bar[i] - k1[i]/a_bar[i]);
			Rx_inv[2][2] = -0.5*(b2*v_bar[i] - k2[i]/a_bar[i]);
			Rx_inv[2][3] = 0.5*b2;
			Rx_inv[3][0] = k1[i]*v_bar[i] - k2[i]*u_bar[i];
			Rx_inv[3][1] = k2[i];
			Rx_inv[3][2] = -k1[i];
			Rx_inv[3][3] = 0.0;

			// Calculate quantities jump
			Q_del[0] = Q1[j*Nx+(i+1)] - Q1[j*Nx+i];
			Q_del[1] = Q2[j*Nx+(i+1)] - Q2[j*Nx+i];
			Q_del[2] = Q3[j*Nx+(i+1)] - Q3[j*Nx+i];
			Q_del[3] = Q4[j*Nx+(i+1)] - Q4[j*Nx+i];

			// Intermedium variables
			for (int k = 0; k<4; k++) {
				sum = 0.0;
				for (int m = 0; m<4; m++) {
					sum += Rx_inv[k][m]*Q_del[m];
				}
				alp[k][i] = sum;
			}
		}

		// BC
		alp_L[0] = alp[0][0];
		alp_L[1] = alp[1][0];
		alp_L[2] = alp[2][0];
		alp_L[3] = alp[3][0];
		alp_R[0] = alp[0][Nx-2];
		alp_R[1] = alp[1][Nx-2];
		alp_R[2] = alp[2][Nx-2];
		alp_R[3] = alp[3][Nx-2];

		// Calculate flux
		for (int i = 0; i<Nx-1; i++) {
			// Contravariant velocity
			Ug = dxgdx_x[j*Nx+i]*u_bar[i] + dxgdy_x[j*Nx+i]*v_bar[i];
			Vg = dygdx_x[j*Nx+i]*u_bar[i] + dygdy_x[j*Nx+i]*v_bar[i];
			// Eigenvalues lam
			lam[0] = Ug - a_bar[i]*xgyg_sq[i];
			lam[1] = Ug;
			lam[2] = Ug + a_bar[i]*xgyg_sq[i];
			lam[3] = Ug;
			// Eigenvector R
			Rx[0][0] = 1.0;
			Rx[0][1] = 1.0;
			Rx[0][2] = 1.0;
			Rx[0][3] = 0.0;
			Rx[1][0] = u_bar[i] - k1[i]*a_bar[i];
			Rx[1][1] = u_bar[i];
			Rx[1][2] = u_bar[i] + k1[i]*a_bar[i];
			Rx[1][3] = k2[i];
			Rx[2][0] = v_bar[i] - k2[i]*a_bar[i];
			Rx[2][1] = v_bar[i];
			Rx[2][2] = v_bar[i] + k2[i]*a_bar[i];
			Rx[2][3] = -k1[i];
			Rx[3][0] = H_bar[i] - (k1[i]*u_bar[i] + k2[i]*v_bar[i])*a_bar[i];
			Rx[3][1] = 0.5*(u_bar[i]*u_bar[i] + v_bar[i]*v_bar[i]);
			Rx[3][2] = H_bar[i] + (k1[i]*u_bar[i] + k2[i]*v_bar[i])*a_bar[i];
			Rx[3][3] = -k1[i]*v_bar[i] + k2[i]*u_bar[i];
			// Entropy fix parameter
		//	lam_ref = max(fabs(lam[0]), max(fabs(lam[1]), fabs(lam[2])));
			temp = dxgdx_x[j*Nx+i]*dxgdx_x[j*Nx+i] + dxgdy_x[j*Nx+i]*dxgdy_x[j*Nx+i] \
				 + dygdx_x[j*Nx+i]*dygdx_x[j*Nx+i] + dygdy_x[j*Nx+i]*dygdy_x[j*Nx+i];
			lam_ref = fabs(Ug) + fabs(Vg) + a_bar[i]*sqrt(temp);
			del_x = del*lam_ref;

			// Flux limiter
			if (i==0)
				for (int k = 0; k<4; k++) {
					if (k==0 || k==2)
						q_limit[k] = minmod(alp_L[k], alp[k][i], alp[k][i+1]);
					else
						q_limit[k] = minmod2(2.0*alp_L[k], 2.0*alp[k][i], 2.0*alp[k][i+1], \
											 0.5*(alp_L[k] + alp[k][i+1]));
				}
			else if (i==Nx-2)
				for (int k = 0; k<4; k++) {
					if (k==0 || k==2)
						q_limit[k] = minmod(alp[k][i-1], alp[k][i], alp_R[k]);
					else
						q_limit[k] = minmod2(2.0*alp[k][i-1], 2.0*alp[k][i], 2.0*alp_R[k], \
											 0.5*(alp[k][i-1] + alp_R[k]));
				}
			else
				for (int k = 0; k<4; k++) {
					if (k==0 || k==2)
						q_limit[k] = minmod(alp[k][i-1], alp[k][i], alp[k][i+1]);
					else
						q_limit[k] = minmod2(2.0*alp[k][i-1], 2.0*alp[k][i], 2.0*alp[k][i+1], \
											 0.5*(alp[k][i-1] + alp[k][i+1]));
				}
			for (int k=0; k<4; k++) {
				// FOU
				//q_limit[k] = 0.0;
			}

			// phi
			for (int k = 0; k<4; k++) {
				// Entropy fix
				if (fabs(lam[k])>=del_x)
					psi = fabs(lam[k]);
				else
					psi = (lam[k]*lam[k] + del_x*del_x)/(2.0*del_x);
				phi[k] = -((dt/dxg)*lam[k]*lam[k]*q_limit[k] + psi*(alp[k][i] - q_limit[k]));
			}

			// Calculate flux
			cal_WQ2E(i, j, E_L);
			cal_WQ2E(i+1, j, E_R);
			cal_WQ2F(i, j, F_L);
			cal_WQ2F(i+1, j, F_R);
			/// First element
			sum = 0.0;
			for (int k = 0; k<4; k++) {
				sum += Rx[0][k]*phi[k];
			}
			E1[j*(Nx-1)+i] = 0.5*((dxgdx_x[j*Nx+i]/Jac_x[j*Nx+i])*(E_L[0] + E_R[0]) \
								  + (dxgdy_x[j*Nx+i]/Jac_x[j*Nx+i])*(F_L[0] + F_R[0])\
								  + sum/Jac_x[j*Nx+i]);
			/// Second element
			sum = 0.0;
			for (int k = 0; k<4; k++) {
				sum += Rx[1][k]*phi[k];
			}
			E2[j*(Nx-1)+i] = 0.5*((dxgdx_x[j*Nx+i]/Jac_x[j*Nx+i])*(E_L[1] + E_R[1]) \
								  + (dxgdy_x[j*Nx+i]/Jac_x[j*Nx+i])*(F_L[1] + F_R[1])\
								  + sum/Jac_x[j*Nx+i]);
			/// Third element
			sum = 0.0;
			for (int k = 0; k<4; k++) {
				sum += Rx[2][k]*phi[k];
			}
			E3[j*(Nx-1)+i] = 0.5*((dxgdx_x[j*Nx+i]/Jac_x[j*Nx+i])*(E_L[2] + E_R[2]) \
								  + (dxgdy_x[j*Nx+i]/Jac_x[j*Nx+i])*(F_L[2] + F_R[2])\
								  + sum/Jac_x[j*Nx+i]);
			/// fourth element
			sum = 0.0;
			for (int k = 0; k<4; k++) {
				sum += Rx[3][k]*phi[k];
			}
			E4[j*(Nx-1)+i] = 0.5*((dxgdx_x[j*Nx+i]/Jac_x[j*Nx+i])*(E_L[3] + E_R[3]) \
								  + (dxgdy_x[j*Nx+i]/Jac_x[j*Nx+i])*(F_L[3] + F_R[3])\
								  + sum/Jac_x[j*Nx+i]);
		}
	}
	// Release memory
	delete u_bar;
	delete v_bar;
	delete a_bar;
	delete H_bar;
	delete k1;
	delete k2;
	delete xgyg_sq;
	for (int k = 0; k<4; k++)
		delete alp[k];
	delete alp;
}


void cal_Flux_y() {
	double temp, temp2, sum;	// for sqrt test, and sum
	double D, *u_bar, *v_bar, *a_bar, *H_bar, H_B, H_T;
	double Ug, Vg;
	double b1, b2;
	double Ry[4][4], Ry_inv[4][4];
	double lam[4];
	double **alp, alp_T[4], alp_B[4], Q_del[4];	// quantity jump
	double q_limit[4];	// Flux limiter
	double phi[4], psi;
	double F_B[4], F_T[4], E_B[4], E_T[4];	// Flux on the left and right hand sides
	double del_y, lam_ref;
	double *k1, *k2, *xgyg_sq;

	// Memory Allocation
	u_bar = new double[Ny-1];
	v_bar = new double[Ny-1];
	a_bar = new double[Ny-1];
	H_bar = new double[Ny-1];
	k1 = new double[Ny-1];
	k2 = new double[Ny-1];
	xgyg_sq = new double[Ny-1];
	alp = new double*[4];
	for (int k = 0; k<4; k++) {
		alp[k] = new double[Ny-1];
	}

	for (int i = 1; i<Nx-1; i++) {
		// Calculate quantity jump
		for (int j = 0; j<Ny-1; j++) {
			// Roe average
			/// sqrt for D
			temp = W1[(j+1)*Nx+i]/W1[j*Nx+i];
			if (temp>=0.0)
				D = sqrt(temp);
			else {
				cout << "domain error in D, j = " << endl;
				output();
				system("pause");
				exit(-1);
			}

			u_bar[j] = (W2[j*Nx+i] + D*W2[(j+1)*Nx+i])/(1.0 + D);
			v_bar[j] = (W3[j*Nx+i] + D*W3[(j+1)*Nx+i])/(1.0 + D);
			H_B = (Q4[j*Nx+i] + W4[j*Nx+i])/W1[j*Nx+i];
			H_T = (Q4[(j+1)*Nx+i] + W4[(j+1)*Nx+i])/W1[(j+1)*Nx+i];
			H_bar[j] = (H_B + D*H_T)/(1.0 + D);
			/// Roe average a
			temp = (gamma - 1.0)*(H_bar[j] - 0.5*(u_bar[j]*u_bar[j] + v_bar[j]*v_bar[j]));
			//temp2 = min(gamma*W4[j*Nx+i]/W1[j*Nx+i], gamma*W4[(j+1)*Nx+i]/W1[(j+1)*Nx+i]);
			//temp = max(temp, temp2);
			if (temp>=0.0)
				a_bar[j] = sqrt(temp);
			else {
				cout << "domain error in a_bar, j = " << endl;
				output();
				system("pause");
				exit(-1);
			}
			/// b1, b2
			b2 = (gamma - 1.0)/(a_bar[j]*a_bar[j]);
			b1 = 0.5*b2*(u_bar[j]*u_bar[j] + v_bar[j]*v_bar[j]);
			xgyg_sq[j] = sqrt(dygdx_y[j*Nx+i]*dygdx_y[j*Nx+i] + dygdy_y[j*Nx+i]*dygdy_y[j*Nx+i]);
			k1[j] = dygdx_y[j*Nx+i]/xgyg_sq[j];
			k2[j] = dygdy_y[j*Nx+i]/xgyg_sq[j];
			// Inverse of eigenvector
			Ry_inv[0][0] = 0.5*(b1 + k1[j]*u_bar[j]/a_bar[j] + k2[j]*v_bar[j]/a_bar[j]);
			Ry_inv[0][1] = -0.5*(b2*u_bar[j] + k1[j]/a_bar[j]);
			Ry_inv[0][2] = -0.5*(b2*v_bar[j] + k2[j]/a_bar[j]);
			Ry_inv[0][3] = 0.5*b2;
			Ry_inv[1][0] = 1.0-b1;
			Ry_inv[1][1] = b2*u_bar[j];
			Ry_inv[1][2] = b2*v_bar[j];
			Ry_inv[1][3] = -b2;
			Ry_inv[2][0] = 0.5*(b1 - k1[j]*u_bar[j]/a_bar[j] - k2[j]*v_bar[j]/a_bar[j]);
			Ry_inv[2][1] = -0.5*(b2*u_bar[j] - k1[j]/a_bar[j]);
			Ry_inv[2][2] = -0.5*(b2*v_bar[j] - k2[j]/a_bar[j]);
			Ry_inv[2][3] = 0.5*b2;
			Ry_inv[3][0] = k1[j]*v_bar[j] - k2[j]*u_bar[j];
			Ry_inv[3][1] = k2[j];
			Ry_inv[3][2] = -k1[j];
			Ry_inv[3][3] = 0.0;

			// Calculate quantities jump
			Q_del[0] = Q1[(j+1)*Nx+i] - Q1[j*Nx+i];
			Q_del[1] = Q2[(j+1)*Nx+i] - Q2[j*Nx+i];
			Q_del[2] = Q3[(j+1)*Nx+i] - Q3[j*Nx+i];
			Q_del[3] = Q4[(j+1)*Nx+i] - Q4[j*Nx+i];

			for (int k = 0; k<4; k++) {
				sum = 0.0;
				for (int m = 0; m<4; m++) {
					sum += Ry_inv[k][m]*Q_del[m];
				}
				alp[k][j] = sum;
			}
		}

		// BC
		alp_B[0] = alp[0][0];
		alp_B[1] = alp[1][0];
		alp_B[2] = alp[2][0];
		alp_B[3] = alp[3][0];
		alp_T[0] = alp[0][Ny-2];
		alp_T[1] = alp[1][Ny-2];
		alp_T[2] = alp[2][Ny-2];
		alp_T[3] = alp[3][Ny-2];

		// Calculate flux
		for (int j = 0; j<Ny-1; j++) {
			// Contravariant velocity
			Ug = dxgdx_y[j*Nx+i]*u_bar[j] + dxgdy_y[j*Nx+i]*v_bar[j];
			Vg = dygdx_y[j*Nx+i]*u_bar[j] + dygdy_y[j*Nx+i]*v_bar[j];
			// Eigenvalues lam
			lam[0] = Vg - a_bar[j]*xgyg_sq[j];
			lam[1] = Vg;
			lam[2] = Vg + a_bar[j]*xgyg_sq[j];
			lam[3] = Vg;
			// Eigenvector R
			Ry[0][0] = 1.0;
			Ry[0][1] = 1.0;
			Ry[0][2] = 1.0;
			Ry[0][3] = 0.0;
			Ry[1][0] = u_bar[j] - k1[j]*a_bar[j];
			Ry[1][1] = u_bar[j];
			Ry[1][2] = u_bar[j] + k1[j]*a_bar[j];
			Ry[1][3] = k2[j];
			Ry[2][0] = v_bar[j] - k2[j]*a_bar[j];
			Ry[2][1] = v_bar[j];
			Ry[2][2] = v_bar[j] + k2[j]*a_bar[j];
			Ry[2][3] = -k1[j];
			Ry[3][0] = H_bar[j] - (k1[j]*u_bar[j] + k2[j]*v_bar[j])*a_bar[j];
			Ry[3][1] = 0.5*(u_bar[j]*u_bar[j] + v_bar[j]*v_bar[j]);
			Ry[3][2] = H_bar[j] + (k1[j]*u_bar[j] + k2[j]*v_bar[j])*a_bar[j];
			Ry[3][3] = -k1[j]*v_bar[j] + k2[j]*u_bar[j];
			// Entropy fix parameter
		//	lam_ref = max(fabs(lam[0]), max(fabs(lam[1]), fabs(lam[2])));
			temp = dxgdx_y[j*Nx+i]*dxgdx_y[j*Nx+i] + dxgdy_y[j*Nx+i]*dxgdy_y[j*Nx+i] \
				 + dygdx_y[j*Nx+i]*dygdx_y[j*Nx+i] + dygdy_y[j*Nx+i]*dygdy_y[j*Nx+i];
			lam_ref = fabs(Ug) + fabs(Vg) + a_bar[j]*sqrt(temp);
			del_y = del*lam_ref;

			// Flux limiter
			if (j==0)
				for (int k = 0; k<4; k++) {
					if (k==0 || k==2)
						q_limit[k] = minmod(alp_B[k], alp[k][j], alp[k][j+1]);
					else
						q_limit[k] = minmod2(2.0*alp_B[k], 2.0*alp[k][j], 2.0*alp[k][j+1], \
											 0.5*(alp_B[k] + alp[k][j+1]));
				}
			else if (j==Ny-2)
				for (int k = 0; k<4; k++) {
					if (k==0 || k==2)
						q_limit[k] = minmod(alp[k][j-1], alp[k][j], alp_T[k]);
					else
						q_limit[k] = minmod2(2.0*alp[k][j-1], 2.0*alp[k][j], 2.0*alp_T[k], \
											 0.5*(alp[k][j-1] + alp_T[k]));
				}
			else
				for (int k = 0; k<4; k++) {
					if (k==0 || k==2)
						q_limit[k] = minmod(alp[k][j-1], alp[k][j], alp[k][j+1]);
					else
						q_limit[k] = minmod2(2.0*alp[k][j-1], 2.0*alp[k][j], 2.0*alp[k][j+1], \
											 0.5*(alp[k][j-1] + alp[k][j+1]));
				}
			for (int k=0; k<4; k++) {
				// FOU
				//q_limit[k] = 0.0;
			}

			// phi
			for (int k = 0; k<4; k++) {
				// Entropy fix
				if (fabs(lam[k])>=del_y)
					psi = fabs(lam[k]);
				else
					psi = (lam[k]*lam[k] + del_y*del_y)/(2.0*del_y);
				phi[k] = -((dt/dyg)*lam[k]*lam[k]*q_limit[k] + psi*(alp[k][j] - q_limit[k]));
			}
			// Calculate flux
			cal_WQ2E(i, j, E_B);
			cal_WQ2E(i, j+1, E_T);
			cal_WQ2F(i, j, F_B);
			cal_WQ2F(i, j+1, F_T);

			/// First element
			sum = 0.0;
			for (int k = 0; k<4; k++) {
				sum += Ry[0][k]*phi[k];
			}
			F1[j*Nx+i] = 0.5*((dygdx_y[j*Nx+i]/Jac_y[j*Nx+i])*(E_B[0] + E_T[0]) \
							  + (dygdy_y[j*Nx+i]/Jac_y[j*Nx+i])*(F_B[0] + F_T[0])\
							  + sum/Jac_y[j*Nx+i]);
			/// Second element
			sum = 0.0;
			for (int k = 0; k<4; k++) {
				sum += Ry[1][k]*phi[k];
			}
			F2[j*Nx+i] = 0.5*((dygdx_y[j*Nx+i]/Jac_y[j*Nx+i])*(E_B[1] + E_T[1]) \
							  + (dygdy_y[j*Nx+i]/Jac_y[j*Nx+i])*(F_B[1] + F_T[1])\
							  + sum/Jac_y[j*Nx+i]);
			/// Third element
			sum = 0.0;
			for (int k = 0; k<4; k++) {
				sum += Ry[2][k]*phi[k];
			}
			F3[j*Nx+i] = 0.5*((dygdx_y[j*Nx+i]/Jac_y[j*Nx+i])*(E_B[2] + E_T[2]) \
							  + (dygdy_y[j*Nx+i]/Jac_y[j*Nx+i])*(F_B[2] + F_T[2])\
							  + sum/Jac_y[j*Nx+i]);
			/// fourth element
			sum = 0.0;
			for (int k = 0; k<4; k++) {
				sum += Ry[3][k]*phi[k];
			}
			F4[j*Nx+i] = 0.5*((dygdx_y[j*Nx+i]/Jac_y[j*Nx+i])*(E_B[3] + E_T[3]) \
							  + (dygdy_y[j*Nx+i]/Jac_y[j*Nx+i])*(F_B[3] + F_T[3])\
							  + sum/Jac_y[j*Nx+i]);
		}
	}

	// Release memory
	delete u_bar;
	delete v_bar;
	delete a_bar;
	delete H_bar;
	delete k1;
	delete k2;
	delete xgyg_sq;
	for (int k = 0; k<4; k++)
		delete alp[k];
	delete alp;
}


void cal_Quan_x() {
	for (int j = 1; j<Ny-1; j++) {
		for (int i = 1; i<Nx-1; i++) {
			// Remember E[j*(Nx-1)+i]
			Q1[j*Nx+i] = Q1[j*Nx+i] - (dt/dxg)*(E1[j*(Nx-1)+i] - E1[j*(Nx-1)+(i-1)])*Jac_c[j*Nx+i];
			Q2[j*Nx+i] = Q2[j*Nx+i] - (dt/dxg)*(E2[j*(Nx-1)+i] - E2[j*(Nx-1)+(i-1)])*Jac_c[j*Nx+i];
			Q3[j*Nx+i] = Q3[j*Nx+i] - (dt/dxg)*(E3[j*(Nx-1)+i] - E3[j*(Nx-1)+(i-1)])*Jac_c[j*Nx+i];
			Q4[j*Nx+i] = Q4[j*Nx+i] - (dt/dxg)*(E4[j*(Nx-1)+i] - E4[j*(Nx-1)+(i-1)])*Jac_c[j*Nx+i];
		}
	}
}


void cal_Quan_y() {
	for (int i = 1; i<Nx-1; i++) {
		for (int j = 1; j<Ny-1; j++) {
			Q1[j*Nx+i] = Q1[j*Nx+i] - (dt/dyg)*(F1[j*Nx+i] - F1[(j-1)*Nx+i])*Jac_c[j*Nx+i];
			Q2[j*Nx+i] = Q2[j*Nx+i] - (dt/dyg)*(F2[j*Nx+i] - F2[(j-1)*Nx+i])*Jac_c[j*Nx+i];
			Q3[j*Nx+i] = Q3[j*Nx+i] - (dt/dyg)*(F3[j*Nx+i] - F3[(j-1)*Nx+i])*Jac_c[j*Nx+i];
			Q4[j*Nx+i] = Q4[j*Nx+i] - (dt/dyg)*(F4[j*Nx+i] - F4[(j-1)*Nx+i])*Jac_c[j*Nx+i];
		}
	}
}


void cal_Prim() {
	for (int j = 1; j<Ny-1; j++) {
		for (int i = 1; i<Nx-1; i++) {
			cal_Q2W(i, j);
		}
	}
}


void cal_W2Q(int i, int j) {
	double e, V_sq;
	Q1[j*Nx+i] = W1[j*Nx+i];			// Q1 = rho
	Q2[j*Nx+i] = W1[j*Nx+i]*W2[j*Nx+i];	// Q2 = rho*u
	Q3[j*Nx+i] = W1[j*Nx+i]*W3[j*Nx+i];	// Q3 = rho*v
	e = W4[j*Nx+i]/((gamma - 1.0)*W1[j*Nx+i]);	// Specific internal energy e = P/((gamma-1)*rho)
	V_sq = W2[j*Nx+i]*W2[j*Nx+i] + W3[j*Nx+i]*W3[j*Nx+i];	// V^2 = u^2 + v^2
	Q4[j*Nx+i] = W1[j*Nx+i]*(0.5*V_sq + e);	// Q4 = Total energy per unit vol.
											// = E = rho*(0.5*V^2 + e)
}


void cal_WQ2E(int i, int j, double* E_LR) {
	E_LR[0] = Q2[j*Nx+i];							// rho*u
	E_LR[1] = Q2[j*Nx+i]*W2[j*Nx+i] + W4[j*Nx+i];	// rho*u*u + p
	E_LR[2] = W1[j*Nx+i]*W2[j*Nx+i]*W3[j*Nx+i];		// rho*u*v
	E_LR[3] = (Q4[j*Nx+i] + W4[j*Nx+i])*W2[j*Nx+i];	// (Et + p)*u
}


void cal_WQ2F(int i, int j, double* F_LR) {
	F_LR[0] = Q3[j*Nx+i];							// rho*v
	F_LR[1] = W1[j*Nx+i]*W2[j*Nx+i]*W3[j*Nx+i];		// rho*u*v
	F_LR[2] = Q3[j*Nx+i]*W3[j*Nx+i] + W4[j*Nx+i];	// rho*v*v + p
	F_LR[3] = (Q4[j*Nx+i] + W4[j*Nx+i])*W3[j*Nx+i];	// (Et + p)*v
}


void cal_Q2W(int i, int j) {
	double e, V_sq;
	W1[j*Nx+i] = Q1[j*Nx+i];
	W2[j*Nx+i] = Q2[j*Nx+i]/Q1[j*Nx+i];
	W3[j*Nx+i] = Q3[j*Nx+i]/Q1[j*Nx+i];
	V_sq = W2[j*Nx+i]*W2[j*Nx+i] + W3[j*Nx+i]*W3[j*Nx+i];
	e = Q4[j*Nx+i]/W1[j*Nx+i] - 0.5*V_sq;
	W4[j*Nx+i] = e*(gamma - 1.0)*W1[j*Nx+i];
	Tem[j*Nx+i] = W4[j*Nx+i]/(Rgas*W1[j*Nx+i]);
}


double minmod(double a, double b, double c) {
	// take sign of a
	double s;
	s = (a>=0.0) ? 1.0 : -1.0;
	return s*max(0.0, min(s*a, min(s*b, s*c)));
}


double minmod2(double a, double b, double c, double d) {
	double s;
	s = (a>=0.0) ? 1.0 : -1.0;
	return s*max(0.0, min(s*a, min(s*b, min(s*c, s*d))));
}


void output_grid() {
	ofstream File_grid;
	File_grid.open("Grid.txt");
	File_grid << "zone i = " << Nx << " j = " << Ny << " DATAPACKING = POINT" << endl;
	for (int j=0; j<Ny; j++)
		for (int i=0; i<Nx; i++)
			File_grid << X[j*Nx+i] << "\t" << Y[j*Nx+i] << "\t" << Jac_c[j*Nx+i]  << endl;
	File_grid.close();
}


void output() {
	// Output
	ofstream File1, File2, File3, File4, File5;
	File1.open("Density.txt");
	File2.open("Vel_u.txt");
	File3.open("Vel_v.txt");
	File4.open("Pressure.txt");
	File5.open("Temperature.txt");
	File1 << "zone i = " << Nx << " j = " << Ny << " DATAPACKING = POINT" << endl;
	File2 << "zone i = " << Nx << " j = " << Ny << " DATAPACKING = POINT" << endl;
	File3 << "zone i = " << Nx << " j = " << Ny << " DATAPACKING = POINT" << endl;
	File4 << "zone i = " << Nx << " j = " << Ny << " DATAPACKING = POINT" << endl;
	File5 << "zone i = " << Nx << " j = " << Ny << " DATAPACKING = POINT" << endl;
	for (int j = 0; j<Ny; j++) {
		for (int i=0; i<Nx; i++) {
			File1 << X[j*Nx+i] << "\t" << Y[j*Nx+i] << "\t" << W1[j*Nx+i] << endl;
			File2 << X[j*Nx+i] << "\t" << Y[j*Nx+i] << "\t" << W2[j*Nx+i] << endl;
			File3 << X[j*Nx+i] << "\t" << Y[j*Nx+i] << "\t" << W3[j*Nx+i] << endl;
			File4 << X[j*Nx+i] << "\t" << Y[j*Nx+i] << "\t" << W4[j*Nx+i] << endl;
			File5 << X[j*Nx+i] << "\t" << Y[j*Nx+i] << "\t" << Tem[j*Nx+i] << endl;
		}
	}
	File1.close();
	File2.close();
	File3.close();
	File4.close();
	File5.close();
}