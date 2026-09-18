/*
 * Copyright (c) The acados authors.
 *
 * This file is part of acados.
 *
 * The 2-Clause BSD License
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.;
 */


// standard
#include <stdio.h>
#include <stdlib.h>
// acados
#include "acados/utils/print.h"
#include "acados/utils/math.h"
#include "acados_c/sim_interface.h"
#include "acados_sim_solver_full_3d_biped_mpc_mc_body.h"

#define NX     FULL_3D_BIPED_MPC_MC_BODY_NX
#define NZ     FULL_3D_BIPED_MPC_MC_BODY_NZ
#define NU     FULL_3D_BIPED_MPC_MC_BODY_NU
#define NP     FULL_3D_BIPED_MPC_MC_BODY_NP


int main()
{
    int status = 0;
    full_3d_biped_mpc_mc_body_sim_solver_capsule *capsule = full_3d_biped_mpc_mc_body_acados_sim_solver_create_capsule();
    status = full_3d_biped_mpc_mc_body_acados_sim_create(capsule);

    if (status)
    {
        printf("acados_create() returned status %d. Exiting.\n", status);
        exit(1);
    }

    sim_config *acados_sim_config = full_3d_biped_mpc_mc_body_acados_get_sim_config(capsule);
    sim_in *acados_sim_in = full_3d_biped_mpc_mc_body_acados_get_sim_in(capsule);
    sim_out *acados_sim_out = full_3d_biped_mpc_mc_body_acados_get_sim_out(capsule);
    void *acados_sim_dims = full_3d_biped_mpc_mc_body_acados_get_sim_dims(capsule);

    // initial condition
    double x_current[NX];
    x_current[0] = 0.0;
    x_current[1] = 0.0;
    x_current[2] = 0.0;
    x_current[3] = 0.0;
    x_current[4] = 0.0;
    x_current[5] = 0.0;
    x_current[6] = 0.0;
    x_current[7] = 0.0;
    x_current[8] = 0.0;
    x_current[9] = 0.0;
    x_current[10] = 0.0;
    x_current[11] = 0.0;

  
    x_current[0] = -0.2273513162563358;
    x_current[1] = -0.002955961642958354;
    x_current[2] = 0.28577725814472527;
    x_current[3] = 0.0000009548597640753636;
    x_current[4] = 0.9078564226423383;
    x_current[5] = 0.03083277055654422;
    x_current[6] = -0.000002654080581054199;
    x_current[7] = 0.0000035047165125688396;
    x_current[8] = -0.000004647193857430093;
    x_current[9] = 0.00006283450476653046;
    x_current[10] = 0.000004796013334089356;
    x_current[11] = -0.00004740848535901047;
    
  


    // initial value for control input
    double u0[NU];
    u0[0] = 0.0;
    u0[1] = 0.0;
    u0[2] = 0.0;
    u0[3] = 0.0;
    u0[4] = 0.0;
    u0[5] = 0.0;
    u0[6] = 0.0;
    u0[7] = 0.0;
    u0[8] = 0.0;
    u0[9] = 0.0;
    // set parameters
    double p[NP];
    p[0] = 22.27;
    p[1] = 9.81;
    p[2] = 0.527;
    p[3] = 0.432;
    p[4] = 0.4;
    p[5] = -0.048;
    p[6] = 0.15;
    p[7] = 0.0019;
    p[8] = -0.048;
    p[9] = -0.15;
    p[10] = 0.0019;
    p[11] = 0.15;
    p[12] = -0.15;

    full_3d_biped_mpc_mc_body_acados_sim_update_params(capsule, p, NP);
  

  


    int n_sim_steps = 3;
    // solve ocp in loop
    for (int ii = 0; ii < n_sim_steps; ii++)
    {
        // set inputs
        sim_in_set(acados_sim_config, acados_sim_dims,
            acados_sim_in, "x", x_current);
        sim_in_set(acados_sim_config, acados_sim_dims,
            acados_sim_in, "u", u0);

        // solve
        status = full_3d_biped_mpc_mc_body_acados_sim_solve(capsule);
        if (status != ACADOS_SUCCESS)
        {
            printf("acados_solve() failed with status %d.\n", status);
        }

        // get outputs
        sim_out_get(acados_sim_config, acados_sim_dims,
               acados_sim_out, "x", x_current);

    

        // print solution
        printf("\nx_current, %d\n", ii);
        for (int jj = 0; jj < NX; jj++)
        {
            printf("%e\n", x_current[jj]);
        }
    }

    printf("\nPerformed %d simulation steps with acados integrator successfully.\n\n", n_sim_steps);

    // free solver
    status = full_3d_biped_mpc_mc_body_acados_sim_free(capsule);
    if (status) {
        printf("full_3d_biped_mpc_mc_body_acados_sim_free() returned status %d. \n", status);
    }

    full_3d_biped_mpc_mc_body_acados_sim_solver_free_capsule(capsule);

    return status;
}
