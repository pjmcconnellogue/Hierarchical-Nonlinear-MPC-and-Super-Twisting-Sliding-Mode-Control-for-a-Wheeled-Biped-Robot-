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

#ifndef ACADOS_SOLVER_full_3d_biped_mpc_mc_body_H_
#define ACADOS_SOLVER_full_3d_biped_mpc_mc_body_H_

#include "acados/utils/types.h"

#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"

#define FULL_3D_BIPED_MPC_MC_BODY_NX     12
#define FULL_3D_BIPED_MPC_MC_BODY_NZ     0
#define FULL_3D_BIPED_MPC_MC_BODY_NU     10
#define FULL_3D_BIPED_MPC_MC_BODY_NP     13
#define FULL_3D_BIPED_MPC_MC_BODY_NP_GLOBAL     0
#define FULL_3D_BIPED_MPC_MC_BODY_NBX    0
#define FULL_3D_BIPED_MPC_MC_BODY_NBX0   12
#define FULL_3D_BIPED_MPC_MC_BODY_NBU    10
#define FULL_3D_BIPED_MPC_MC_BODY_NSBX   0
#define FULL_3D_BIPED_MPC_MC_BODY_NSBU   0
#define FULL_3D_BIPED_MPC_MC_BODY_NSH    2
#define FULL_3D_BIPED_MPC_MC_BODY_NSH0   0
#define FULL_3D_BIPED_MPC_MC_BODY_NSG    0
#define FULL_3D_BIPED_MPC_MC_BODY_NSPHI  0
#define FULL_3D_BIPED_MPC_MC_BODY_NSHN   0
#define FULL_3D_BIPED_MPC_MC_BODY_NSGN   0
#define FULL_3D_BIPED_MPC_MC_BODY_NSPHIN 0
#define FULL_3D_BIPED_MPC_MC_BODY_NSPHI0 0
#define FULL_3D_BIPED_MPC_MC_BODY_NSBXN  0
#define FULL_3D_BIPED_MPC_MC_BODY_NS     2
#define FULL_3D_BIPED_MPC_MC_BODY_NS0    0
#define FULL_3D_BIPED_MPC_MC_BODY_NSN    0
#define FULL_3D_BIPED_MPC_MC_BODY_NG     0
#define FULL_3D_BIPED_MPC_MC_BODY_NBXN   0
#define FULL_3D_BIPED_MPC_MC_BODY_NGN    0
#define FULL_3D_BIPED_MPC_MC_BODY_NY0    22
#define FULL_3D_BIPED_MPC_MC_BODY_NY     22
#define FULL_3D_BIPED_MPC_MC_BODY_NYN    12
#define FULL_3D_BIPED_MPC_MC_BODY_N      10
#define FULL_3D_BIPED_MPC_MC_BODY_NH     2
#define FULL_3D_BIPED_MPC_MC_BODY_NHN    0
#define FULL_3D_BIPED_MPC_MC_BODY_NH0    0
#define FULL_3D_BIPED_MPC_MC_BODY_NPHI0  0
#define FULL_3D_BIPED_MPC_MC_BODY_NPHI   0
#define FULL_3D_BIPED_MPC_MC_BODY_NPHIN  0
#define FULL_3D_BIPED_MPC_MC_BODY_NR     0

#ifdef __cplusplus
extern "C" {
#endif


// ** capsule for solver data **
typedef struct full_3d_biped_mpc_mc_body_solver_capsule
{
    // acados objects
    ocp_nlp_in *nlp_in;
    ocp_nlp_out *nlp_out;
    ocp_nlp_out *sens_out;
    ocp_nlp_solver *nlp_solver;
    void *nlp_opts;
    ocp_nlp_plan_t *nlp_solver_plan;
    ocp_nlp_config *nlp_config;
    ocp_nlp_dims *nlp_dims;

    // number of expected runtime parameters
    unsigned int nlp_np;

    /* external functions */

    // dynamics

    external_function_external_param_casadi *impl_dae_fun;
    external_function_external_param_casadi *impl_dae_fun_jac_x_xdot_z;
    external_function_external_param_casadi *impl_dae_jac_x_xdot_u_z;
    external_function_external_param_casadi *impl_dae_jac_p;




    // cost






    // constraints
    external_function_external_param_casadi *nl_constr_h_fun_jac;
    external_function_external_param_casadi *nl_constr_h_fun;









} full_3d_biped_mpc_mc_body_solver_capsule;

ACADOS_SYMBOL_EXPORT full_3d_biped_mpc_mc_body_solver_capsule * full_3d_biped_mpc_mc_body_acados_create_capsule(void);
ACADOS_SYMBOL_EXPORT int full_3d_biped_mpc_mc_body_acados_free_capsule(full_3d_biped_mpc_mc_body_solver_capsule *capsule);

ACADOS_SYMBOL_EXPORT int full_3d_biped_mpc_mc_body_acados_create(full_3d_biped_mpc_mc_body_solver_capsule * capsule);

ACADOS_SYMBOL_EXPORT int full_3d_biped_mpc_mc_body_acados_reset(full_3d_biped_mpc_mc_body_solver_capsule* capsule, int reset_qp_solver_mem);

/**
 * Generic version of full_3d_biped_mpc_mc_body_acados_create which allows to use a different number of shooting intervals than
 * the number used for code generation. If new_time_steps=NULL and n_time_steps matches the number used for code
 * generation, the time-steps from code generation is used.
 */
ACADOS_SYMBOL_EXPORT int full_3d_biped_mpc_mc_body_acados_create_with_discretization(full_3d_biped_mpc_mc_body_solver_capsule * capsule, int n_time_steps, double* new_time_steps);
/**
 * Update the time step vector. Number N must be identical to the currently set number of shooting nodes in the
 * nlp_solver_plan. Returns 0 if no error occurred and a otherwise a value other than 0.
 */
ACADOS_SYMBOL_EXPORT int full_3d_biped_mpc_mc_body_acados_update_time_steps(full_3d_biped_mpc_mc_body_solver_capsule * capsule, int N, double* new_time_steps);
/**
 * This function is used for updating an already initialized solver with a different number of qp_cond_N.
 */
ACADOS_SYMBOL_EXPORT int full_3d_biped_mpc_mc_body_acados_update_qp_solver_cond_N(full_3d_biped_mpc_mc_body_solver_capsule * capsule, int qp_solver_cond_N);
ACADOS_SYMBOL_EXPORT int full_3d_biped_mpc_mc_body_acados_update_params(full_3d_biped_mpc_mc_body_solver_capsule * capsule, int stage, double *value, int np);
ACADOS_SYMBOL_EXPORT int full_3d_biped_mpc_mc_body_acados_update_params_sparse(full_3d_biped_mpc_mc_body_solver_capsule * capsule, int stage, int *idx, double *p, int n_update);
ACADOS_SYMBOL_EXPORT int full_3d_biped_mpc_mc_body_acados_set_p_global_and_precompute_dependencies(full_3d_biped_mpc_mc_body_solver_capsule* capsule, double* data, int data_len);

ACADOS_SYMBOL_EXPORT int full_3d_biped_mpc_mc_body_acados_solve(full_3d_biped_mpc_mc_body_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT int full_3d_biped_mpc_mc_body_acados_setup_qp_matrices_and_factorize(full_3d_biped_mpc_mc_body_solver_capsule* capsule);



ACADOS_SYMBOL_EXPORT int full_3d_biped_mpc_mc_body_acados_free(full_3d_biped_mpc_mc_body_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT void full_3d_biped_mpc_mc_body_acados_print_stats(full_3d_biped_mpc_mc_body_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT int full_3d_biped_mpc_mc_body_acados_custom_update(full_3d_biped_mpc_mc_body_solver_capsule* capsule, double* data, int data_len);

ACADOS_SYMBOL_EXPORT ocp_nlp_in *full_3d_biped_mpc_mc_body_acados_get_nlp_in(full_3d_biped_mpc_mc_body_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_out *full_3d_biped_mpc_mc_body_acados_get_nlp_out(full_3d_biped_mpc_mc_body_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_out *full_3d_biped_mpc_mc_body_acados_get_sens_out(full_3d_biped_mpc_mc_body_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_solver *full_3d_biped_mpc_mc_body_acados_get_nlp_solver(full_3d_biped_mpc_mc_body_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_config *full_3d_biped_mpc_mc_body_acados_get_nlp_config(full_3d_biped_mpc_mc_body_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT void *full_3d_biped_mpc_mc_body_acados_get_nlp_opts(full_3d_biped_mpc_mc_body_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_dims *full_3d_biped_mpc_mc_body_acados_get_nlp_dims(full_3d_biped_mpc_mc_body_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_plan_t *full_3d_biped_mpc_mc_body_acados_get_nlp_plan(full_3d_biped_mpc_mc_body_solver_capsule * capsule);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // ACADOS_SOLVER_full_3d_biped_mpc_mc_body_H_
