#ifndef OPENSBLIBLOCK00_KERNEL_H
#define OPENSBLIBLOCK00_KERNEL_H

#include "math.h"

// Declaration of global constants
extern int restart;
extern int iter;
extern int stage;
extern double tstart;
extern double DRP_filt;
extern double Delta0block0;
extern double Delta1block0;
extern double Delta2block0;
extern int HDF5_timing;
extern double Minf;
extern double Pr;
extern double Re;
extern int block0np0;
extern int block0np1;
extern int block0np2;
extern double dt;
extern int filter_frequency;
extern double gama;
extern double inv2Delta0block0;
extern double inv2Delta1block0;
extern double inv2Delta2block0;
extern double inv2Minf;
extern double invDelta0block0;
extern double invDelta1block0;
extern double invDelta2block0;
extern double invPr;
extern double invRe;
extern double inv_gamma_m1;
extern int niter;
extern double simulation_time;
extern int start_iter;
extern int write_output_file;

void opensbliblock00Kernel039(ACC<double> &rhoE_B0, ACC<double> &rho_B0, ACC<double> &rhou0_B0, ACC<double> &rhou1_B0,
ACC<double> &rhou2_B0, const int *idx)
{
   double p = 0.0;
   double r = 0.0;
   double u0 = 0.0;
   double u1 = 0.0;
   double u2 = 0.0;
   double x0 = 0.0;
   double x1 = 0.0;
   double x2 = 0.0;
   x0 = Delta0block0*idx[0];

   x1 = Delta1block0*idx[1];

   x2 = Delta2block0*idx[2];

   u0 = cos(x1)*cos(x2)*sin(x0);

   u1 = -cos(x0)*cos(x2)*sin(x1);

   u2 = 0.0;

   p = (2.0 + cos(2.0*x2))*(0.0625*cos(2.0*x0) + 0.0625*cos(2.0*x1)) + 1.0/((Minf*Minf)*gama);

   r = gama*p*(Minf*Minf);

   rho_B0(0,0,0) = r;

   rhou0_B0(0,0,0) = r*u0;

   rhou1_B0(0,0,0) = r*u1;

   rhou2_B0(0,0,0) = r*u2;

   rhoE_B0(0,0,0) = p/(-1 + gama) + 0.5*r*((u0*u0) + (u1*u1) + (u2*u2));

}

void opensbliblock00Kernel008(const ACC<double> &rho_B0, const ACC<double> &rhou0_B0, ACC<double> &u0_B0)
{
   u0_B0(0,0,0) = rhou0_B0(0,0,0)/rho_B0(0,0,0);

}

void opensbliblock00Kernel010(const ACC<double> &rho_B0, const ACC<double> &rhou1_B0, ACC<double> &u1_B0)
{
   u1_B0(0,0,0) = rhou1_B0(0,0,0)/rho_B0(0,0,0);

}

void opensbliblock00Kernel012(const ACC<double> &rho_B0, const ACC<double> &rhou2_B0, ACC<double> &u2_B0)
{
   u2_B0(0,0,0) = rhou2_B0(0,0,0)/rho_B0(0,0,0);

}

 void opensbliblock00Kernel019(const ACC<double> &rhoE_B0, const ACC<double> &rho_B0, const ACC<double> &u0_B0, const
ACC<double> &u1_B0, const ACC<double> &u2_B0, ACC<double> &p_B0)
{
    p_B0(0,0,0) = (-1 + gama)*(-(1.0/2.0)*(u0_B0(0,0,0)*u0_B0(0,0,0))*rho_B0(0,0,0) -
      (1.0/2.0)*(u1_B0(0,0,0)*u1_B0(0,0,0))*rho_B0(0,0,0) - (1.0/2.0)*(u2_B0(0,0,0)*u2_B0(0,0,0))*rho_B0(0,0,0) +
      rhoE_B0(0,0,0));

}

void opensbliblock00Kernel025(const ACC<double> &p_B0, const ACC<double> &rho_B0, ACC<double> &T_B0)
{
   T_B0(0,0,0) = (Minf*Minf)*gama*p_B0(0,0,0)/rho_B0(0,0,0);

}

void opensbliblock00Kernel007(const ACC<double> &u0_B0, ACC<double> &wk0_B0)
{
    wk0_B0(0,0,0) = (-(2.0/3.0)*u0_B0(-1,0,0) - (1.0/12.0)*u0_B0(2,0,0) + ((1.0/12.0))*u0_B0(-2,0,0) +
      ((2.0/3.0))*u0_B0(1,0,0))*invDelta0block0;

}

void opensbliblock00Kernel009(const ACC<double> &u1_B0, ACC<double> &wk1_B0)
{
    wk1_B0(0,0,0) = (-(2.0/3.0)*u1_B0(-1,0,0) - (1.0/12.0)*u1_B0(2,0,0) + ((1.0/12.0))*u1_B0(-2,0,0) +
      ((2.0/3.0))*u1_B0(1,0,0))*invDelta0block0;

}

void opensbliblock00Kernel011(const ACC<double> &u2_B0, ACC<double> &wk2_B0)
{
    wk2_B0(0,0,0) = (-(2.0/3.0)*u2_B0(-1,0,0) - (1.0/12.0)*u2_B0(2,0,0) + ((1.0/12.0))*u2_B0(-2,0,0) +
      ((2.0/3.0))*u2_B0(1,0,0))*invDelta0block0;

}

void opensbliblock00Kernel013(const ACC<double> &u0_B0, ACC<double> &wk3_B0)
{
    wk3_B0(0,0,0) = (-(2.0/3.0)*u0_B0(0,-1,0) - (1.0/12.0)*u0_B0(0,2,0) + ((1.0/12.0))*u0_B0(0,-2,0) +
      ((2.0/3.0))*u0_B0(0,1,0))*invDelta1block0;

}

void opensbliblock00Kernel014(const ACC<double> &u1_B0, ACC<double> &wk4_B0)
{
    wk4_B0(0,0,0) = (-(2.0/3.0)*u1_B0(0,-1,0) - (1.0/12.0)*u1_B0(0,2,0) + ((1.0/12.0))*u1_B0(0,-2,0) +
      ((2.0/3.0))*u1_B0(0,1,0))*invDelta1block0;

}

void opensbliblock00Kernel015(const ACC<double> &u2_B0, ACC<double> &wk5_B0)
{
    wk5_B0(0,0,0) = (-(2.0/3.0)*u2_B0(0,-1,0) - (1.0/12.0)*u2_B0(0,2,0) + ((1.0/12.0))*u2_B0(0,-2,0) +
      ((2.0/3.0))*u2_B0(0,1,0))*invDelta1block0;

}

void opensbliblock00Kernel016(const ACC<double> &u0_B0, ACC<double> &wk6_B0)
{
    wk6_B0(0,0,0) = (-(2.0/3.0)*u0_B0(0,0,-1) - (1.0/12.0)*u0_B0(0,0,2) + ((1.0/12.0))*u0_B0(0,0,-2) +
      ((2.0/3.0))*u0_B0(0,0,1))*invDelta2block0;

}

void opensbliblock00Kernel017(const ACC<double> &u1_B0, ACC<double> &wk7_B0)
{
    wk7_B0(0,0,0) = (-(2.0/3.0)*u1_B0(0,0,-1) - (1.0/12.0)*u1_B0(0,0,2) + ((1.0/12.0))*u1_B0(0,0,-2) +
      ((2.0/3.0))*u1_B0(0,0,1))*invDelta2block0;

}

void opensbliblock00Kernel018(const ACC<double> &u2_B0, ACC<double> &wk8_B0)
{
    wk8_B0(0,0,0) = (-(2.0/3.0)*u2_B0(0,0,-1) - (1.0/12.0)*u2_B0(0,0,2) + ((1.0/12.0))*u2_B0(0,0,-2) +
      ((2.0/3.0))*u2_B0(0,0,1))*invDelta2block0;

}

 void opensbliblock00Kernel031(const ACC<double> &p_B0, const ACC<double> &rhoE_B0, const ACC<double> &rho_B0, const
ACC<double> &rhou0_B0, const ACC<double> &rhou1_B0, const ACC<double> &rhou2_B0, const ACC<double> &u0_B0, const
ACC<double> &u1_B0, const ACC<double> &u2_B0, const ACC<double> &wk0_B0, const ACC<double> &wk1_B0, const ACC<double>
&wk2_B0, const ACC<double> &wk3_B0, const ACC<double> &wk4_B0, const ACC<double> &wk5_B0, const ACC<double> &wk6_B0,
const ACC<double> &wk7_B0, const ACC<double> &wk8_B0, ACC<double> &Residual0_B0, ACC<double> &Residual1_B0, ACC<double>
&Residual2_B0, ACC<double> &Residual3_B0, ACC<double> &Residual4_B0)
{
   double d1_inv_rhoErho_dx = 0.0;
   double d1_inv_rhoErho_dy = 0.0;
   double d1_inv_rhoErho_dz = 0.0;
   double d1_p_dx = 0.0;
   double d1_p_dy = 0.0;
   double d1_p_dz = 0.0;
   double d1_pu0_dx = 0.0;
   double d1_pu1_dy = 0.0;
   double d1_pu2_dz = 0.0;
   double d1_rhoEu0_dx = 0.0;
   double d1_rhoEu1_dy = 0.0;
   double d1_rhoEu2_dz = 0.0;
   double d1_rhou0_dx = 0.0;
   double d1_rhou0u0_dx = 0.0;
   double d1_rhou0u1_dy = 0.0;
   double d1_rhou0u2_dz = 0.0;
   double d1_rhou1_dy = 0.0;
   double d1_rhou1u0_dx = 0.0;
   double d1_rhou1u1_dy = 0.0;
   double d1_rhou1u2_dz = 0.0;
   double d1_rhou2_dz = 0.0;
   double d1_rhou2u0_dx = 0.0;
   double d1_rhou2u1_dy = 0.0;
   double d1_rhou2u2_dz = 0.0;
    d1_p_dx = (-(2.0/3.0)*p_B0(-1,0,0) - (1.0/12.0)*p_B0(2,0,0) + ((1.0/12.0))*p_B0(-2,0,0) +
      ((2.0/3.0))*p_B0(1,0,0))*invDelta0block0;

    d1_pu0_dx = (-(2.0/3.0)*p_B0(-1,0,0)*u0_B0(-1,0,0) - (1.0/12.0)*p_B0(2,0,0)*u0_B0(2,0,0) +
      ((1.0/12.0))*p_B0(-2,0,0)*u0_B0(-2,0,0) + ((2.0/3.0))*p_B0(1,0,0)*u0_B0(1,0,0))*invDelta0block0;

    d1_rhoEu0_dx = (-(2.0/3.0)*u0_B0(-1,0,0)*rhoE_B0(-1,0,0) - (1.0/12.0)*u0_B0(2,0,0)*rhoE_B0(2,0,0) +
      ((1.0/12.0))*u0_B0(-2,0,0)*rhoE_B0(-2,0,0) + ((2.0/3.0))*u0_B0(1,0,0)*rhoE_B0(1,0,0))*invDelta0block0;

    d1_inv_rhoErho_dx = (-(2.0/3.0)*rhoE_B0(-1,0,0)/rho_B0(-1,0,0) - (1.0/12.0)*rhoE_B0(2,0,0)/rho_B0(2,0,0) +
      ((1.0/12.0))*rhoE_B0(-2,0,0)/rho_B0(-2,0,0) + ((2.0/3.0))*rhoE_B0(1,0,0)/rho_B0(1,0,0))*invDelta0block0;

    d1_rhou0_dx = (-(2.0/3.0)*rhou0_B0(-1,0,0) - (1.0/12.0)*rhou0_B0(2,0,0) + ((1.0/12.0))*rhou0_B0(-2,0,0) +
      ((2.0/3.0))*rhou0_B0(1,0,0))*invDelta0block0;

    d1_rhou0u0_dx = (-(2.0/3.0)*u0_B0(-1,0,0)*rhou0_B0(-1,0,0) - (1.0/12.0)*u0_B0(2,0,0)*rhou0_B0(2,0,0) +
      ((1.0/12.0))*u0_B0(-2,0,0)*rhou0_B0(-2,0,0) + ((2.0/3.0))*u0_B0(1,0,0)*rhou0_B0(1,0,0))*invDelta0block0;

    d1_rhou1u0_dx = (-(2.0/3.0)*u0_B0(-1,0,0)*rhou1_B0(-1,0,0) - (1.0/12.0)*u0_B0(2,0,0)*rhou1_B0(2,0,0) +
      ((1.0/12.0))*u0_B0(-2,0,0)*rhou1_B0(-2,0,0) + ((2.0/3.0))*u0_B0(1,0,0)*rhou1_B0(1,0,0))*invDelta0block0;

    d1_rhou2u0_dx = (-(2.0/3.0)*u0_B0(-1,0,0)*rhou2_B0(-1,0,0) - (1.0/12.0)*u0_B0(2,0,0)*rhou2_B0(2,0,0) +
      ((1.0/12.0))*u0_B0(-2,0,0)*rhou2_B0(-2,0,0) + ((2.0/3.0))*u0_B0(1,0,0)*rhou2_B0(1,0,0))*invDelta0block0;

    d1_p_dy = (-(2.0/3.0)*p_B0(0,-1,0) - (1.0/12.0)*p_B0(0,2,0) + ((1.0/12.0))*p_B0(0,-2,0) +
      ((2.0/3.0))*p_B0(0,1,0))*invDelta1block0;

    d1_pu1_dy = (-(2.0/3.0)*p_B0(0,-1,0)*u1_B0(0,-1,0) - (1.0/12.0)*p_B0(0,2,0)*u1_B0(0,2,0) +
      ((1.0/12.0))*p_B0(0,-2,0)*u1_B0(0,-2,0) + ((2.0/3.0))*p_B0(0,1,0)*u1_B0(0,1,0))*invDelta1block0;

    d1_rhoEu1_dy = (-(2.0/3.0)*u1_B0(0,-1,0)*rhoE_B0(0,-1,0) - (1.0/12.0)*u1_B0(0,2,0)*rhoE_B0(0,2,0) +
      ((1.0/12.0))*u1_B0(0,-2,0)*rhoE_B0(0,-2,0) + ((2.0/3.0))*u1_B0(0,1,0)*rhoE_B0(0,1,0))*invDelta1block0;

    d1_inv_rhoErho_dy = (-(2.0/3.0)*rhoE_B0(0,-1,0)/rho_B0(0,-1,0) - (1.0/12.0)*rhoE_B0(0,2,0)/rho_B0(0,2,0) +
      ((1.0/12.0))*rhoE_B0(0,-2,0)/rho_B0(0,-2,0) + ((2.0/3.0))*rhoE_B0(0,1,0)/rho_B0(0,1,0))*invDelta1block0;

    d1_rhou0u1_dy = (-(2.0/3.0)*u1_B0(0,-1,0)*rhou0_B0(0,-1,0) - (1.0/12.0)*u1_B0(0,2,0)*rhou0_B0(0,2,0) +
      ((1.0/12.0))*u1_B0(0,-2,0)*rhou0_B0(0,-2,0) + ((2.0/3.0))*u1_B0(0,1,0)*rhou0_B0(0,1,0))*invDelta1block0;

    d1_rhou1_dy = (-(2.0/3.0)*rhou1_B0(0,-1,0) - (1.0/12.0)*rhou1_B0(0,2,0) + ((1.0/12.0))*rhou1_B0(0,-2,0) +
      ((2.0/3.0))*rhou1_B0(0,1,0))*invDelta1block0;

    d1_rhou1u1_dy = (-(2.0/3.0)*u1_B0(0,-1,0)*rhou1_B0(0,-1,0) - (1.0/12.0)*u1_B0(0,2,0)*rhou1_B0(0,2,0) +
      ((1.0/12.0))*u1_B0(0,-2,0)*rhou1_B0(0,-2,0) + ((2.0/3.0))*u1_B0(0,1,0)*rhou1_B0(0,1,0))*invDelta1block0;

    d1_rhou2u1_dy = (-(2.0/3.0)*u1_B0(0,-1,0)*rhou2_B0(0,-1,0) - (1.0/12.0)*u1_B0(0,2,0)*rhou2_B0(0,2,0) +
      ((1.0/12.0))*u1_B0(0,-2,0)*rhou2_B0(0,-2,0) + ((2.0/3.0))*u1_B0(0,1,0)*rhou2_B0(0,1,0))*invDelta1block0;

    d1_p_dz = (-(2.0/3.0)*p_B0(0,0,-1) - (1.0/12.0)*p_B0(0,0,2) + ((1.0/12.0))*p_B0(0,0,-2) +
      ((2.0/3.0))*p_B0(0,0,1))*invDelta2block0;

    d1_pu2_dz = (-(2.0/3.0)*p_B0(0,0,-1)*u2_B0(0,0,-1) - (1.0/12.0)*p_B0(0,0,2)*u2_B0(0,0,2) +
      ((1.0/12.0))*p_B0(0,0,-2)*u2_B0(0,0,-2) + ((2.0/3.0))*p_B0(0,0,1)*u2_B0(0,0,1))*invDelta2block0;

    d1_rhoEu2_dz = (-(2.0/3.0)*u2_B0(0,0,-1)*rhoE_B0(0,0,-1) - (1.0/12.0)*u2_B0(0,0,2)*rhoE_B0(0,0,2) +
      ((1.0/12.0))*u2_B0(0,0,-2)*rhoE_B0(0,0,-2) + ((2.0/3.0))*u2_B0(0,0,1)*rhoE_B0(0,0,1))*invDelta2block0;

    d1_inv_rhoErho_dz = (-(2.0/3.0)*rhoE_B0(0,0,-1)/rho_B0(0,0,-1) - (1.0/12.0)*rhoE_B0(0,0,2)/rho_B0(0,0,2) +
      ((1.0/12.0))*rhoE_B0(0,0,-2)/rho_B0(0,0,-2) + ((2.0/3.0))*rhoE_B0(0,0,1)/rho_B0(0,0,1))*invDelta2block0;

    d1_rhou0u2_dz = (-(2.0/3.0)*u2_B0(0,0,-1)*rhou0_B0(0,0,-1) - (1.0/12.0)*u2_B0(0,0,2)*rhou0_B0(0,0,2) +
      ((1.0/12.0))*u2_B0(0,0,-2)*rhou0_B0(0,0,-2) + ((2.0/3.0))*u2_B0(0,0,1)*rhou0_B0(0,0,1))*invDelta2block0;

    d1_rhou1u2_dz = (-(2.0/3.0)*u2_B0(0,0,-1)*rhou1_B0(0,0,-1) - (1.0/12.0)*u2_B0(0,0,2)*rhou1_B0(0,0,2) +
      ((1.0/12.0))*u2_B0(0,0,-2)*rhou1_B0(0,0,-2) + ((2.0/3.0))*u2_B0(0,0,1)*rhou1_B0(0,0,1))*invDelta2block0;

    d1_rhou2_dz = (-(2.0/3.0)*rhou2_B0(0,0,-1) - (1.0/12.0)*rhou2_B0(0,0,2) + ((1.0/12.0))*rhou2_B0(0,0,-2) +
      ((2.0/3.0))*rhou2_B0(0,0,1))*invDelta2block0;

    d1_rhou2u2_dz = (-(2.0/3.0)*u2_B0(0,0,-1)*rhou2_B0(0,0,-1) - (1.0/12.0)*u2_B0(0,0,2)*rhou2_B0(0,0,2) +
      ((1.0/12.0))*u2_B0(0,0,-2)*rhou2_B0(0,0,-2) + ((2.0/3.0))*u2_B0(0,0,1)*rhou2_B0(0,0,1))*invDelta2block0;

   Residual0_B0(0,0,0) = -d1_rhou0_dx - d1_rhou1_dy - d1_rhou2_dz;

    Residual1_B0(0,0,0) = -d1_p_dx - (1.0/2.0)*d1_rhou0u0_dx - (1.0/2.0)*d1_rhou0u1_dy - (1.0/2.0)*d1_rhou0u2_dz -
      (1.0/2.0)*u0_B0(0,0,0)*d1_rhou0_dx - (1.0/2.0)*u0_B0(0,0,0)*d1_rhou1_dy - (1.0/2.0)*u0_B0(0,0,0)*d1_rhou2_dz -
      (1.0/2.0)*wk0_B0(0,0,0)*rhou0_B0(0,0,0) - (1.0/2.0)*wk3_B0(0,0,0)*rhou1_B0(0,0,0) -
      (1.0/2.0)*wk6_B0(0,0,0)*rhou2_B0(0,0,0);

    Residual2_B0(0,0,0) = -d1_p_dy - (1.0/2.0)*d1_rhou1u0_dx - (1.0/2.0)*d1_rhou1u1_dy - (1.0/2.0)*d1_rhou1u2_dz -
      (1.0/2.0)*(d1_rhou0_dx + d1_rhou1_dy + d1_rhou2_dz)*u1_B0(0,0,0) - (1.0/2.0)*wk1_B0(0,0,0)*rhou0_B0(0,0,0) -
      (1.0/2.0)*wk4_B0(0,0,0)*rhou1_B0(0,0,0) - (1.0/2.0)*wk7_B0(0,0,0)*rhou2_B0(0,0,0);

    Residual3_B0(0,0,0) = -d1_p_dz - (1.0/2.0)*d1_rhou2u0_dx - (1.0/2.0)*d1_rhou2u1_dy - (1.0/2.0)*d1_rhou2u2_dz -
      (1.0/2.0)*(d1_rhou0_dx + d1_rhou1_dy + d1_rhou2_dz)*u2_B0(0,0,0) - (1.0/2.0)*wk2_B0(0,0,0)*rhou0_B0(0,0,0) -
      (1.0/2.0)*wk5_B0(0,0,0)*rhou1_B0(0,0,0) - (1.0/2.0)*wk8_B0(0,0,0)*rhou2_B0(0,0,0);

    Residual4_B0(0,0,0) = -d1_pu0_dx - d1_pu1_dy - d1_pu2_dz - (1.0/2.0)*d1_rhoEu0_dx - (1.0/2.0)*d1_rhoEu1_dy -
      (1.0/2.0)*d1_rhoEu2_dz - (1.0/2.0)*rhou0_B0(0,0,0)*d1_inv_rhoErho_dx - (1.0/2.0)*rhou1_B0(0,0,0)*d1_inv_rhoErho_dy
      - (1.0/2.0)*rhou2_B0(0,0,0)*d1_inv_rhoErho_dz - (1.0/2.0)*(d1_rhou0_dx + d1_rhou1_dy +
      d1_rhou2_dz)*rhoE_B0(0,0,0)/rho_B0(0,0,0);

}

 void opensbliblock00Kernel032(const ACC<double> &T_B0, const ACC<double> &u0_B0, const ACC<double> &u1_B0, const
ACC<double> &u2_B0, const ACC<double> &wk0_B0, const ACC<double> &wk1_B0, const ACC<double> &wk2_B0, const ACC<double>
&wk3_B0, const ACC<double> &wk4_B0, const ACC<double> &wk5_B0, const ACC<double> &wk6_B0, const ACC<double> &wk7_B0,
const ACC<double> &wk8_B0, ACC<double> &Residual1_B0, ACC<double> &Residual2_B0, ACC<double> &Residual3_B0, ACC<double>
&Residual4_B0)
{
   double d1_wk0_dy = 0.0;
   double d1_wk0_dz = 0.0;
   double d1_wk1_dy = 0.0;
   double d1_wk2_dz = 0.0;
   double d1_wk4_dz = 0.0;
   double d1_wk5_dz = 0.0;
   double d2_T_dx = 0.0;
   double d2_T_dy = 0.0;
   double d2_T_dz = 0.0;
   double d2_u0_dx = 0.0;
   double d2_u0_dy = 0.0;
   double d2_u0_dz = 0.0;
   double d2_u1_dx = 0.0;
   double d2_u1_dy = 0.0;
   double d2_u1_dz = 0.0;
   double d2_u2_dx = 0.0;
   double d2_u2_dy = 0.0;
   double d2_u2_dz = 0.0;
    d2_T_dx = (-(5.0/2.0)*T_B0(0,0,0) - (1.0/12.0)*T_B0(-2,0,0) - (1.0/12.0)*T_B0(2,0,0) + ((4.0/3.0))*T_B0(1,0,0) +
      ((4.0/3.0))*T_B0(-1,0,0))*inv2Delta0block0;

    d2_u0_dx = (-(5.0/2.0)*u0_B0(0,0,0) - (1.0/12.0)*u0_B0(-2,0,0) - (1.0/12.0)*u0_B0(2,0,0) + ((4.0/3.0))*u0_B0(1,0,0)
      + ((4.0/3.0))*u0_B0(-1,0,0))*inv2Delta0block0;

    d2_u1_dx = (-(5.0/2.0)*u1_B0(0,0,0) - (1.0/12.0)*u1_B0(-2,0,0) - (1.0/12.0)*u1_B0(2,0,0) + ((4.0/3.0))*u1_B0(1,0,0)
      + ((4.0/3.0))*u1_B0(-1,0,0))*inv2Delta0block0;

    d2_u2_dx = (-(5.0/2.0)*u2_B0(0,0,0) - (1.0/12.0)*u2_B0(-2,0,0) - (1.0/12.0)*u2_B0(2,0,0) + ((4.0/3.0))*u2_B0(1,0,0)
      + ((4.0/3.0))*u2_B0(-1,0,0))*inv2Delta0block0;

    d2_T_dy = (-(5.0/2.0)*T_B0(0,0,0) - (1.0/12.0)*T_B0(0,-2,0) - (1.0/12.0)*T_B0(0,2,0) + ((4.0/3.0))*T_B0(0,1,0) +
      ((4.0/3.0))*T_B0(0,-1,0))*inv2Delta1block0;

    d2_u0_dy = (-(5.0/2.0)*u0_B0(0,0,0) - (1.0/12.0)*u0_B0(0,-2,0) - (1.0/12.0)*u0_B0(0,2,0) + ((4.0/3.0))*u0_B0(0,1,0)
      + ((4.0/3.0))*u0_B0(0,-1,0))*inv2Delta1block0;

    d2_u1_dy = (-(5.0/2.0)*u1_B0(0,0,0) - (1.0/12.0)*u1_B0(0,-2,0) - (1.0/12.0)*u1_B0(0,2,0) + ((4.0/3.0))*u1_B0(0,1,0)
      + ((4.0/3.0))*u1_B0(0,-1,0))*inv2Delta1block0;

    d2_u2_dy = (-(5.0/2.0)*u2_B0(0,0,0) - (1.0/12.0)*u2_B0(0,-2,0) - (1.0/12.0)*u2_B0(0,2,0) + ((4.0/3.0))*u2_B0(0,1,0)
      + ((4.0/3.0))*u2_B0(0,-1,0))*inv2Delta1block0;

    d1_wk0_dy = (-(2.0/3.0)*wk0_B0(0,-1,0) - (1.0/12.0)*wk0_B0(0,2,0) + ((1.0/12.0))*wk0_B0(0,-2,0) +
      ((2.0/3.0))*wk0_B0(0,1,0))*invDelta1block0;

    d1_wk1_dy = (-(2.0/3.0)*wk1_B0(0,-1,0) - (1.0/12.0)*wk1_B0(0,2,0) + ((1.0/12.0))*wk1_B0(0,-2,0) +
      ((2.0/3.0))*wk1_B0(0,1,0))*invDelta1block0;

    d2_T_dz = (-(5.0/2.0)*T_B0(0,0,0) - (1.0/12.0)*T_B0(0,0,-2) - (1.0/12.0)*T_B0(0,0,2) + ((4.0/3.0))*T_B0(0,0,1) +
      ((4.0/3.0))*T_B0(0,0,-1))*inv2Delta2block0;

    d2_u0_dz = (-(5.0/2.0)*u0_B0(0,0,0) - (1.0/12.0)*u0_B0(0,0,-2) - (1.0/12.0)*u0_B0(0,0,2) + ((4.0/3.0))*u0_B0(0,0,1)
      + ((4.0/3.0))*u0_B0(0,0,-1))*inv2Delta2block0;

    d2_u1_dz = (-(5.0/2.0)*u1_B0(0,0,0) - (1.0/12.0)*u1_B0(0,0,-2) - (1.0/12.0)*u1_B0(0,0,2) + ((4.0/3.0))*u1_B0(0,0,1)
      + ((4.0/3.0))*u1_B0(0,0,-1))*inv2Delta2block0;

    d2_u2_dz = (-(5.0/2.0)*u2_B0(0,0,0) - (1.0/12.0)*u2_B0(0,0,-2) - (1.0/12.0)*u2_B0(0,0,2) + ((4.0/3.0))*u2_B0(0,0,1)
      + ((4.0/3.0))*u2_B0(0,0,-1))*inv2Delta2block0;

    d1_wk0_dz = (-(2.0/3.0)*wk0_B0(0,0,-1) - (1.0/12.0)*wk0_B0(0,0,2) + ((1.0/12.0))*wk0_B0(0,0,-2) +
      ((2.0/3.0))*wk0_B0(0,0,1))*invDelta2block0;

    d1_wk2_dz = (-(2.0/3.0)*wk2_B0(0,0,-1) - (1.0/12.0)*wk2_B0(0,0,2) + ((1.0/12.0))*wk2_B0(0,0,-2) +
      ((2.0/3.0))*wk2_B0(0,0,1))*invDelta2block0;

    d1_wk4_dz = (-(2.0/3.0)*wk4_B0(0,0,-1) - (1.0/12.0)*wk4_B0(0,0,2) + ((1.0/12.0))*wk4_B0(0,0,-2) +
      ((2.0/3.0))*wk4_B0(0,0,1))*invDelta2block0;

    d1_wk5_dz = (-(2.0/3.0)*wk5_B0(0,0,-1) - (1.0/12.0)*wk5_B0(0,0,2) + ((1.0/12.0))*wk5_B0(0,0,-2) +
      ((2.0/3.0))*wk5_B0(0,0,1))*invDelta2block0;

    Residual1_B0(0,0,0) = 1.0*(((1.0/3.0))*d1_wk1_dy + ((1.0/3.0))*d1_wk2_dz + ((4.0/3.0))*d2_u0_dx + d2_u0_dy +
      d2_u0_dz)*invRe + Residual1_B0(0,0,0);

    Residual2_B0(0,0,0) = 1.0*(((1.0/3.0))*d1_wk0_dy + ((1.0/3.0))*d1_wk5_dz + ((4.0/3.0))*d2_u1_dy + d2_u1_dx +
      d2_u1_dz)*invRe + Residual2_B0(0,0,0);

    Residual3_B0(0,0,0) = 1.0*(((1.0/3.0))*d1_wk0_dz + ((1.0/3.0))*d1_wk4_dz + ((4.0/3.0))*d2_u2_dz + d2_u2_dx +
      d2_u2_dy)*invRe + Residual3_B0(0,0,0);

    Residual4_B0(0,0,0) = 1.0*(wk1_B0(0,0,0) + wk3_B0(0,0,0))*invRe*wk1_B0(0,0,0) + 1.0*(wk1_B0(0,0,0) +
      wk3_B0(0,0,0))*invRe*wk3_B0(0,0,0) + 1.0*(wk2_B0(0,0,0) + wk6_B0(0,0,0))*invRe*wk2_B0(0,0,0) + 1.0*(wk2_B0(0,0,0)
      + wk6_B0(0,0,0))*invRe*wk6_B0(0,0,0) + 1.0*(wk5_B0(0,0,0) + wk7_B0(0,0,0))*invRe*wk5_B0(0,0,0) +
      1.0*(wk5_B0(0,0,0) + wk7_B0(0,0,0))*invRe*wk7_B0(0,0,0) + 1.0*(-(2.0/3.0)*wk0_B0(0,0,0) - (2.0/3.0)*wk4_B0(0,0,0)
      + ((4.0/3.0))*wk8_B0(0,0,0))*invRe*wk8_B0(0,0,0) + 1.0*(-(2.0/3.0)*wk0_B0(0,0,0) - (2.0/3.0)*wk8_B0(0,0,0) +
      ((4.0/3.0))*wk4_B0(0,0,0))*invRe*wk4_B0(0,0,0) + 1.0*(-(2.0/3.0)*wk4_B0(0,0,0) - (2.0/3.0)*wk8_B0(0,0,0) +
      ((4.0/3.0))*wk0_B0(0,0,0))*invRe*wk0_B0(0,0,0) + 1.0*(((1.0/3.0))*d1_wk0_dy + ((1.0/3.0))*d1_wk5_dz +
      ((4.0/3.0))*d2_u1_dy + d2_u1_dx + d2_u1_dz)*invRe*u1_B0(0,0,0) + 1.0*(((1.0/3.0))*d1_wk0_dz +
      ((1.0/3.0))*d1_wk4_dz + ((4.0/3.0))*d2_u2_dz + d2_u2_dx + d2_u2_dy)*invRe*u2_B0(0,0,0) +
      1.0*(((1.0/3.0))*d1_wk1_dy + ((1.0/3.0))*d1_wk2_dz + ((4.0/3.0))*d2_u0_dx + d2_u0_dy +
      d2_u0_dz)*invRe*u0_B0(0,0,0) + 1.0*(d2_T_dx + d2_T_dy + d2_T_dz)*invPr*invRe*inv2Minf*inv_gamma_m1 +
      Residual4_B0(0,0,0);

}

 void opensbliblock00Kernel040(const ACC<double> &Residual0_B0, const ACC<double> &Residual1_B0, const ACC<double>
&Residual2_B0, const ACC<double> &Residual3_B0, const ACC<double> &Residual4_B0, ACC<double> &rhoE_B0, ACC<double>
&rhoE_RKold_B0, ACC<double> &rho_B0, ACC<double> &rho_RKold_B0, ACC<double> &rhou0_B0, ACC<double> &rhou0_RKold_B0,
ACC<double> &rhou1_B0, ACC<double> &rhou1_RKold_B0, ACC<double> &rhou2_B0, ACC<double> &rhou2_RKold_B0, const double
*rkA, const double *rkB)
{
   rho_RKold_B0(0,0,0) = rkA[0]*rho_RKold_B0(0,0,0) + dt*Residual0_B0(0,0,0);

   rho_B0(0,0,0) = rkB[0]*rho_RKold_B0(0,0,0) + rho_B0(0,0,0);

   rhou0_RKold_B0(0,0,0) = rkA[0]*rhou0_RKold_B0(0,0,0) + dt*Residual1_B0(0,0,0);

   rhou0_B0(0,0,0) = rkB[0]*rhou0_RKold_B0(0,0,0) + rhou0_B0(0,0,0);

   rhou1_RKold_B0(0,0,0) = rkA[0]*rhou1_RKold_B0(0,0,0) + dt*Residual2_B0(0,0,0);

   rhou1_B0(0,0,0) = rkB[0]*rhou1_RKold_B0(0,0,0) + rhou1_B0(0,0,0);

   rhou2_RKold_B0(0,0,0) = rkA[0]*rhou2_RKold_B0(0,0,0) + dt*Residual3_B0(0,0,0);

   rhou2_B0(0,0,0) = rkB[0]*rhou2_RKold_B0(0,0,0) + rhou2_B0(0,0,0);

   rhoE_RKold_B0(0,0,0) = rkA[0]*rhoE_RKold_B0(0,0,0) + dt*Residual4_B0(0,0,0);

   rhoE_B0(0,0,0) = rkB[0]*rhoE_RKold_B0(0,0,0) + rhoE_B0(0,0,0);

}

 void opensbliblock00Kernel000(ACC<double> &rhoE_RKold_B0, ACC<double> &rho_RKold_B0, ACC<double> &rhou0_RKold_B0,
ACC<double> &rhou1_RKold_B0, ACC<double> &rhou2_RKold_B0)
{
   rho_RKold_B0(0,0,0) = 0.0;

   rhou0_RKold_B0(0,0,0) = 0.0;

   rhou1_RKold_B0(0,0,0) = 0.0;

   rhou2_RKold_B0(0,0,0) = 0.0;

   rhoE_RKold_B0(0,0,0) = 0.0;

}

 void opensbliblock00Kernel001(const ACC<double> &rhoE_B0, const ACC<double> &rho_B0, const ACC<double> &rhou0_B0, const
ACC<double> &rhou1_B0, const ACC<double> &rhou2_B0, ACC<double> &rhoE_RKold_B0, ACC<double> &rho_RKold_B0, ACC<double>
&rhou0_RKold_B0, ACC<double> &rhou1_RKold_B0, ACC<double> &rhou2_RKold_B0)
{
    rho_RKold_B0(0,0,0) = 0.215044884112*rho_B0(0,0,0) + 0.123755948787*rho_B0(-2,0,0) + 0.123755948787*rho_B0(2,0,0) +
      0.018721609157*rho_B0(-4,0,0) + 0.018721609157*rho_B0(4,0,0) - 0.059227575576*rho_B0(-3,0,0) -
      0.059227575576*rho_B0(3,0,0) - 0.002999540835*rho_B0(-5,0,0) - 0.002999540835*rho_B0(5,0,0) -
      0.187772883589*rho_B0(1,0,0) - 0.187772883589*rho_B0(-1,0,0);

    rhou0_RKold_B0(0,0,0) = 0.215044884112*rhou0_B0(0,0,0) + 0.123755948787*rhou0_B0(-2,0,0) +
      0.123755948787*rhou0_B0(2,0,0) + 0.018721609157*rhou0_B0(-4,0,0) + 0.018721609157*rhou0_B0(4,0,0) -
      0.059227575576*rhou0_B0(-3,0,0) - 0.059227575576*rhou0_B0(3,0,0) - 0.002999540835*rhou0_B0(-5,0,0) -
      0.002999540835*rhou0_B0(5,0,0) - 0.187772883589*rhou0_B0(1,0,0) - 0.187772883589*rhou0_B0(-1,0,0);

    rhou1_RKold_B0(0,0,0) = 0.215044884112*rhou1_B0(0,0,0) + 0.123755948787*rhou1_B0(-2,0,0) +
      0.123755948787*rhou1_B0(2,0,0) + 0.018721609157*rhou1_B0(-4,0,0) + 0.018721609157*rhou1_B0(4,0,0) -
      0.059227575576*rhou1_B0(-3,0,0) - 0.059227575576*rhou1_B0(3,0,0) - 0.002999540835*rhou1_B0(-5,0,0) -
      0.002999540835*rhou1_B0(5,0,0) - 0.187772883589*rhou1_B0(1,0,0) - 0.187772883589*rhou1_B0(-1,0,0);

    rhou2_RKold_B0(0,0,0) = 0.215044884112*rhou2_B0(0,0,0) + 0.123755948787*rhou2_B0(-2,0,0) +
      0.123755948787*rhou2_B0(2,0,0) + 0.018721609157*rhou2_B0(-4,0,0) + 0.018721609157*rhou2_B0(4,0,0) -
      0.059227575576*rhou2_B0(-3,0,0) - 0.059227575576*rhou2_B0(3,0,0) - 0.002999540835*rhou2_B0(-5,0,0) -
      0.002999540835*rhou2_B0(5,0,0) - 0.187772883589*rhou2_B0(1,0,0) - 0.187772883589*rhou2_B0(-1,0,0);

    rhoE_RKold_B0(0,0,0) = 0.215044884112*rhoE_B0(0,0,0) + 0.123755948787*rhoE_B0(-2,0,0) +
      0.123755948787*rhoE_B0(2,0,0) + 0.018721609157*rhoE_B0(-4,0,0) + 0.018721609157*rhoE_B0(4,0,0) -
      0.059227575576*rhoE_B0(-3,0,0) - 0.059227575576*rhoE_B0(3,0,0) - 0.002999540835*rhoE_B0(-5,0,0) -
      0.002999540835*rhoE_B0(5,0,0) - 0.187772883589*rhoE_B0(1,0,0) - 0.187772883589*rhoE_B0(-1,0,0);

}

 void opensbliblock00Kernel002(const ACC<double> &rhoE_RKold_B0, const ACC<double> &rho_RKold_B0, const ACC<double>
&rhou0_RKold_B0, const ACC<double> &rhou1_RKold_B0, const ACC<double> &rhou2_RKold_B0, ACC<double> &rhoE_B0, ACC<double>
&rho_B0, ACC<double> &rhou0_B0, ACC<double> &rhou1_B0, ACC<double> &rhou2_B0)
{
   rho_B0(0,0,0) = -DRP_filt*rho_RKold_B0(0,0,0) + rho_B0(0,0,0);

   rhou0_B0(0,0,0) = -DRP_filt*rhou0_RKold_B0(0,0,0) + rhou0_B0(0,0,0);

   rhou1_B0(0,0,0) = -DRP_filt*rhou1_RKold_B0(0,0,0) + rhou1_B0(0,0,0);

   rhou2_B0(0,0,0) = -DRP_filt*rhou2_RKold_B0(0,0,0) + rhou2_B0(0,0,0);

   rhoE_B0(0,0,0) = -DRP_filt*rhoE_RKold_B0(0,0,0) + rhoE_B0(0,0,0);

}

 void opensbliblock00Kernel003(const ACC<double> &rhoE_B0, const ACC<double> &rho_B0, const ACC<double> &rhou0_B0, const
ACC<double> &rhou1_B0, const ACC<double> &rhou2_B0, ACC<double> &rhoE_RKold_B0, ACC<double> &rho_RKold_B0, ACC<double>
&rhou0_RKold_B0, ACC<double> &rhou1_RKold_B0, ACC<double> &rhou2_RKold_B0)
{
    rho_RKold_B0(0,0,0) = 0.215044884112*rho_B0(0,0,0) + 0.123755948787*rho_B0(0,-2,0) + 0.123755948787*rho_B0(0,2,0) +
      0.018721609157*rho_B0(0,-4,0) + 0.018721609157*rho_B0(0,4,0) - 0.059227575576*rho_B0(0,-3,0) -
      0.059227575576*rho_B0(0,3,0) - 0.002999540835*rho_B0(0,-5,0) - 0.002999540835*rho_B0(0,5,0) -
      0.187772883589*rho_B0(0,1,0) - 0.187772883589*rho_B0(0,-1,0);

    rhou0_RKold_B0(0,0,0) = 0.215044884112*rhou0_B0(0,0,0) + 0.123755948787*rhou0_B0(0,-2,0) +
      0.123755948787*rhou0_B0(0,2,0) + 0.018721609157*rhou0_B0(0,-4,0) + 0.018721609157*rhou0_B0(0,4,0) -
      0.059227575576*rhou0_B0(0,-3,0) - 0.059227575576*rhou0_B0(0,3,0) - 0.002999540835*rhou0_B0(0,-5,0) -
      0.002999540835*rhou0_B0(0,5,0) - 0.187772883589*rhou0_B0(0,1,0) - 0.187772883589*rhou0_B0(0,-1,0);

    rhou1_RKold_B0(0,0,0) = 0.215044884112*rhou1_B0(0,0,0) + 0.123755948787*rhou1_B0(0,-2,0) +
      0.123755948787*rhou1_B0(0,2,0) + 0.018721609157*rhou1_B0(0,-4,0) + 0.018721609157*rhou1_B0(0,4,0) -
      0.059227575576*rhou1_B0(0,-3,0) - 0.059227575576*rhou1_B0(0,3,0) - 0.002999540835*rhou1_B0(0,-5,0) -
      0.002999540835*rhou1_B0(0,5,0) - 0.187772883589*rhou1_B0(0,1,0) - 0.187772883589*rhou1_B0(0,-1,0);

    rhou2_RKold_B0(0,0,0) = 0.215044884112*rhou2_B0(0,0,0) + 0.123755948787*rhou2_B0(0,-2,0) +
      0.123755948787*rhou2_B0(0,2,0) + 0.018721609157*rhou2_B0(0,-4,0) + 0.018721609157*rhou2_B0(0,4,0) -
      0.059227575576*rhou2_B0(0,-3,0) - 0.059227575576*rhou2_B0(0,3,0) - 0.002999540835*rhou2_B0(0,-5,0) -
      0.002999540835*rhou2_B0(0,5,0) - 0.187772883589*rhou2_B0(0,1,0) - 0.187772883589*rhou2_B0(0,-1,0);

    rhoE_RKold_B0(0,0,0) = 0.215044884112*rhoE_B0(0,0,0) + 0.123755948787*rhoE_B0(0,-2,0) +
      0.123755948787*rhoE_B0(0,2,0) + 0.018721609157*rhoE_B0(0,-4,0) + 0.018721609157*rhoE_B0(0,4,0) -
      0.059227575576*rhoE_B0(0,-3,0) - 0.059227575576*rhoE_B0(0,3,0) - 0.002999540835*rhoE_B0(0,-5,0) -
      0.002999540835*rhoE_B0(0,5,0) - 0.187772883589*rhoE_B0(0,1,0) - 0.187772883589*rhoE_B0(0,-1,0);

}

 void opensbliblock00Kernel004(const ACC<double> &rhoE_RKold_B0, const ACC<double> &rho_RKold_B0, const ACC<double>
&rhou0_RKold_B0, const ACC<double> &rhou1_RKold_B0, const ACC<double> &rhou2_RKold_B0, ACC<double> &rhoE_B0, ACC<double>
&rho_B0, ACC<double> &rhou0_B0, ACC<double> &rhou1_B0, ACC<double> &rhou2_B0)
{
   rho_B0(0,0,0) = -DRP_filt*rho_RKold_B0(0,0,0) + rho_B0(0,0,0);

   rhou0_B0(0,0,0) = -DRP_filt*rhou0_RKold_B0(0,0,0) + rhou0_B0(0,0,0);

   rhou1_B0(0,0,0) = -DRP_filt*rhou1_RKold_B0(0,0,0) + rhou1_B0(0,0,0);

   rhou2_B0(0,0,0) = -DRP_filt*rhou2_RKold_B0(0,0,0) + rhou2_B0(0,0,0);

   rhoE_B0(0,0,0) = -DRP_filt*rhoE_RKold_B0(0,0,0) + rhoE_B0(0,0,0);

}

 void opensbliblock00Kernel005(const ACC<double> &rhoE_B0, const ACC<double> &rho_B0, const ACC<double> &rhou0_B0, const
ACC<double> &rhou1_B0, const ACC<double> &rhou2_B0, ACC<double> &rhoE_RKold_B0, ACC<double> &rho_RKold_B0, ACC<double>
&rhou0_RKold_B0, ACC<double> &rhou1_RKold_B0, ACC<double> &rhou2_RKold_B0)
{
    rho_RKold_B0(0,0,0) = 0.215044884112*rho_B0(0,0,0) + 0.123755948787*rho_B0(0,0,-2) + 0.123755948787*rho_B0(0,0,2) +
      0.018721609157*rho_B0(0,0,-4) + 0.018721609157*rho_B0(0,0,4) - 0.059227575576*rho_B0(0,0,-3) -
      0.059227575576*rho_B0(0,0,3) - 0.002999540835*rho_B0(0,0,-5) - 0.002999540835*rho_B0(0,0,5) -
      0.187772883589*rho_B0(0,0,1) - 0.187772883589*rho_B0(0,0,-1);

    rhou0_RKold_B0(0,0,0) = 0.215044884112*rhou0_B0(0,0,0) + 0.123755948787*rhou0_B0(0,0,-2) +
      0.123755948787*rhou0_B0(0,0,2) + 0.018721609157*rhou0_B0(0,0,-4) + 0.018721609157*rhou0_B0(0,0,4) -
      0.059227575576*rhou0_B0(0,0,-3) - 0.059227575576*rhou0_B0(0,0,3) - 0.002999540835*rhou0_B0(0,0,-5) -
      0.002999540835*rhou0_B0(0,0,5) - 0.187772883589*rhou0_B0(0,0,1) - 0.187772883589*rhou0_B0(0,0,-1);

    rhou1_RKold_B0(0,0,0) = 0.215044884112*rhou1_B0(0,0,0) + 0.123755948787*rhou1_B0(0,0,-2) +
      0.123755948787*rhou1_B0(0,0,2) + 0.018721609157*rhou1_B0(0,0,-4) + 0.018721609157*rhou1_B0(0,0,4) -
      0.059227575576*rhou1_B0(0,0,-3) - 0.059227575576*rhou1_B0(0,0,3) - 0.002999540835*rhou1_B0(0,0,-5) -
      0.002999540835*rhou1_B0(0,0,5) - 0.187772883589*rhou1_B0(0,0,1) - 0.187772883589*rhou1_B0(0,0,-1);

    rhou2_RKold_B0(0,0,0) = 0.215044884112*rhou2_B0(0,0,0) + 0.123755948787*rhou2_B0(0,0,-2) +
      0.123755948787*rhou2_B0(0,0,2) + 0.018721609157*rhou2_B0(0,0,-4) + 0.018721609157*rhou2_B0(0,0,4) -
      0.059227575576*rhou2_B0(0,0,-3) - 0.059227575576*rhou2_B0(0,0,3) - 0.002999540835*rhou2_B0(0,0,-5) -
      0.002999540835*rhou2_B0(0,0,5) - 0.187772883589*rhou2_B0(0,0,1) - 0.187772883589*rhou2_B0(0,0,-1);

    rhoE_RKold_B0(0,0,0) = 0.215044884112*rhoE_B0(0,0,0) + 0.123755948787*rhoE_B0(0,0,-2) +
      0.123755948787*rhoE_B0(0,0,2) + 0.018721609157*rhoE_B0(0,0,-4) + 0.018721609157*rhoE_B0(0,0,4) -
      0.059227575576*rhoE_B0(0,0,-3) - 0.059227575576*rhoE_B0(0,0,3) - 0.002999540835*rhoE_B0(0,0,-5) -
      0.002999540835*rhoE_B0(0,0,5) - 0.187772883589*rhoE_B0(0,0,1) - 0.187772883589*rhoE_B0(0,0,-1);

}

 void opensbliblock00Kernel006(const ACC<double> &rhoE_RKold_B0, const ACC<double> &rho_RKold_B0, const ACC<double>
&rhou0_RKold_B0, const ACC<double> &rhou1_RKold_B0, const ACC<double> &rhou2_RKold_B0, ACC<double> &rhoE_B0, ACC<double>
&rho_B0, ACC<double> &rhou0_B0, ACC<double> &rhou1_B0, ACC<double> &rhou2_B0)
{
   rho_B0(0,0,0) = -DRP_filt*rho_RKold_B0(0,0,0) + rho_B0(0,0,0);

   rhou0_B0(0,0,0) = -DRP_filt*rhou0_RKold_B0(0,0,0) + rhou0_B0(0,0,0);

   rhou1_B0(0,0,0) = -DRP_filt*rhou1_RKold_B0(0,0,0) + rhou1_B0(0,0,0);

   rhou2_B0(0,0,0) = -DRP_filt*rhou2_RKold_B0(0,0,0) + rhou2_B0(0,0,0);

   rhoE_B0(0,0,0) = -DRP_filt*rhoE_RKold_B0(0,0,0) + rhoE_B0(0,0,0);

}

#endif
