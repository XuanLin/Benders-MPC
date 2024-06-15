#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <iostream>
	
#include <Eigen/Sparse>
#include <unsupported/Eigen/CXX11/Tensor>

#include <string>
#include <list>
#include <vector>
#include <array>
#include <stack>
#include <numeric>
#include <cmath>
#include <map>

#include <ctime>
#include <chrono>

#include "gurobi_c++.h"
#include "Eigen_types.h"

using namespace std;

namespace py = pybind11;

// List of z_sol for single time step
const vector<vector<double>> arr_z{{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                   {0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                   {0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                   {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                   {0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                   {0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                   {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0},
                                   {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0},
                                   {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0},
                                   {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0},
                                   {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};

const int num_sol = arr_z.size();

// ===========================================================================================================================================================
// Dual name structure
template <int nx, int nu, int nz, int nc, int N, int dual_len=(N+1)*nx+N*nc>
struct id_dual{
    vector<string> dual_name;
    int id_x0[nx];
    int id_x_dyn[N][nx];
    int id_xuz[N][nc];

    id_dual(){
        for (int i_x=0; i_x<nx; i_x++){dual_name.push_back("x0_item_" + to_string(i_x));}

        for (int i_n=0; i_n<N; i_n++){
            for (int i_x=0; i_x<nx; i_x++){dual_name.push_back("x_dyn_" + to_string(i_n) + "_item_" + to_string(i_x));}}
            
        for (int i_n=0; i_n<N; i_n++){
            for (int i_c=0; i_c<nc; i_c++){dual_name.push_back("xuz_" + to_string(i_n) + "_item_" + to_string(i_c));}}

        for (int i_x=0; i_x<nx; i_x++){
            ptrdiff_t pos = distance(dual_name.begin(), find(dual_name.begin(), dual_name.end(), "x0_item_" + to_string(i_x)));
            id_x0[i_x] = pos;}
            
        for (int i_n=0; i_n<N; i_n++){
            for (int i_x=0; i_x<nx; i_x++){
                ptrdiff_t pos = distance(dual_name.begin(), find(dual_name.begin(), dual_name.end(), "x_dyn_" + to_string(i_n) + "_item_" + to_string(i_x)));
                id_x_dyn[i_n][i_x] = pos;}}

        for (int i_n=0; i_n<N; i_n++){
            for (int i_c=0; i_c<nc; i_c++){
                ptrdiff_t pos = distance(dual_name.begin(), find(dual_name.begin(), dual_name.end(), "xuz_" + to_string(i_n) + "_item_" + to_string(i_c)));
                id_xuz[i_n][i_c] = pos;}}
    }
};

// ===========================================================================================================================================================
// Binary insertion
void insert_into_cost(list<int>& ret_z, list<int>::iterator lf_ret_z, list<int>::iterator ri_ret_z, 
                      list<double>& ret_obj, list<double>::iterator lf_ret_obj, list<double>::iterator ri_ret_obj, int size, int zz, double cost){

    if (size==0){
        ret_obj.insert(lf_ret_obj, cost); 
        ret_z.insert(lf_ret_z, zz);
        return;
    }
    else if (size==1){
        if (cost > *lf_ret_obj){
            ret_obj.insert(lf_ret_obj, cost); 
            ret_z.insert(lf_ret_z, zz);}
        else {
            ret_obj.insert(next(lf_ret_obj), cost);
            ret_z.insert(next(lf_ret_z), zz);}
        return;
    }
    else if (size==2){
        if (cost > *lf_ret_obj){
            ret_obj.insert(lf_ret_obj, cost);
            ret_z.insert(lf_ret_z, zz);}
        else if (cost > *next(lf_ret_obj)){
            ret_obj.insert(next(lf_ret_obj), cost);
            ret_z.insert(next(lf_ret_z), zz);}
        else{ret_obj.insert(next(next(lf_ret_obj)), cost);
             ret_z.insert(next(next(lf_ret_z)), zz);}
        return;
    }

    int mid=size/2;

    list<double>::iterator pt_mid_obj = lf_ret_obj;
    list<int>::iterator pt_mid_z = lf_ret_z;
    advance(pt_mid_obj, mid);
    advance(pt_mid_z, mid);

    if (*pt_mid_obj == cost){
        ret_obj.insert(pt_mid_obj, cost);
        ret_z.insert(pt_mid_z, zz);}
    else if (cost < *pt_mid_obj){
        insert_into_cost(ret_z, pt_mid_z, ri_ret_z, ret_obj, pt_mid_obj, ri_ret_obj, size-mid, zz, cost);
    }
    else{
        insert_into_cost(ret_z, lf_ret_z, pt_mid_z, ret_obj, lf_ret_obj, pt_mid_obj, mid+1, zz, cost);
    }
}

// ===========================================================================================================================================================
// main GBD class

class def_env
{
    public:
        // Should be in static method
        def_env(){
            env.set("LogFile", "subQP.log");
            env.start();
        }

    protected:
        GRBEnv env = GRBEnv(true);
};

template <int nx, int nu, int nz, int nc, int N, int K_feas, int k_ah, int K_opt, int dual_len=nx*(N+1)+nc*N>
class MLD_GBD : public def_env
{
    public:

        MLD_GBD(){

            Q <<   1.,   0.,   0.,   0.,
                   0.,   1.,   0.,   0.,
                   0.,   0.,  0.2,   0.,
                   0.,   0.,   0.,  0.2;

            R <<  0.1,  0.,
                  0., 0.1;

            Qn <<  1.,   0.,   0.,   0.,
                   0.,   1.,   0.,   0.,
                   0.,   0.,  0.2,   0.,
                   0.,   0.,   0.,  0.2;
                    
            E <<  1., 0., 0.65,   0.,
                  0., 1.,   0., 0.65,
                  0., 0.,   1.,   0.,
                  0., 0.,   0.,   1.;  
            
            F <<   0.,   0.,
                   0.,   0.,
                 0.65,   0.,
                   0., 0.65;
            
            G << 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.;
 
            H1 << 1.,  0.,  0.,  0.,
                 -1.,  0.,  0.,  0.,
                  0.,  1.,  0.,  0.,
                  0., -1.,  0.,  0.,
                  1.,  0.,  0.,  0.,
                 -1.,  0.,  0.,  0.,
                  0.,  1.,  0.,  0.,
                  0., -1.,  0.,  0.,
                  1.,  0.,  0.,  0.,
                 -1.,  0.,  0.,  0.,
                  0.,  1.,  0.,  0.,
                  0., -1.,  0.,  0.,
                  1.,  0.,  0.,  0.,
                 -1.,  0.,  0.,  0.,
                  0.,  1.,  0.,  0.,
                  0., -1.,  0.,  0.,
                  1.,  0.,  0.,  0.,
                 -1.,  0.,  0.,  0.,
                  0.,  1.,  0.,  0.,
                  0., -1.,  0.,  0.,
                  1.,  0.,  0.,  0.,
                 -1.,  0.,  0.,  0.,
                  0.,  1.,  0.,  0.,
                  0., -1.,  0.,  0.,
                  1.,  0.,  0.,  0.,
                 -1.,  0.,  0.,  0.,
                  0.,  1.,  0.,  0.,
                  0., -1.,  0.,  0.,
                  1.,  0.,  0.,  0.,
                 -1.,  0.,  0.,  0.,
                  0.,  1.,  0.,  0.,
                  0., -1.,  0.,  0.,
                  1.,  0.,  0.,  0.,
                 -1.,  0.,  0.,  0.,
                  0.,  1.,  0.,  0.,
                  0., -1.,  0.,  0.,
                  1.,  0.,  0.,  0.,
                 -1.,  0.,  0.,  0.,
                  0.,  1.,  0.,  0.,
                  0., -1.,  0.,  0.,
                  1.,  0.,  0.,  0.,
                 -1.,  0.,  0.,  0.,
                  0.,  1.,  0.,  0.,
                  0., -1.,  0.,  0.,
                  0.,  0.,  0.,  0.,
                  0.,  0.,  0.,  0.,
                  0.,  0.,  0.,  0.,
                  0.,  0.,  0.,  0.,
                  0.,  0.,  0.,  0.,
                  0.,  0.,  0.,  0.,
                  0.,  0.,  0.,  0.,
                  0.,  0.,  0.,  0.,
                  0.,  0.,  1.,  0.,
                  0.,  0., -1.,  0.,
                  0.,  0.,  0.,  1.,
                  0.,  0.,  0., -1.,
                  0.,  0.,  0.,  0.,
                  0.,  0.,  0.,  0.,
                  0.,  0.,  0.,  0.,
                  0.,  0.,  0.,  0.;
            
            H2 << 0.,  0.,
                  0.,  0.,
                  0.,  0.,
                  0.,  0.,
                  0.,  0.,
                  0.,  0.,
                  0.,  0.,
                  0.,  0.,
                  0.,  0.,
                  0.,  0.,
                  0.,  0.,
                  0.,  0.,
                  0.,  0.,
                  0.,  0.,
                  0.,  0.,
                  0.,  0.,
                  0.,  0.,
                  0.,  0.,
                  0.,  0.,
                  0.,  0.,
                  0.,  0.,
                  0.,  0.,
                  0.,  0.,
                  0.,  0.,
                  0.,  0.,
                  0.,  0.,
                  0.,  0.,
                  0.,  0.,
                  0.,  0.,
                  0.,  0.,
                  0.,  0.,
                  0.,  0.,
                  0.,  0.,
                  0.,  0.,
                  0.,  0.,
                  0.,  0.,
                  0.,  0.,
                  0.,  0.,
                  0.,  0.,
                  0.,  0.,
                  0.,  0.,
                  0.,  0.,
                  0.,  0.,
                  0.,  0.,
                  1.,  0.,
                 -1.,  0.,
                  0.,  1.,
                  0., -1.,
                  1.,  0.,
                 -1.,  0.,
                  0.,  1.,
                  0., -1.,
                  0.,  0.,
                  0.,  0.,
                  0.,  0.,
                  0.,  0.,
                  1.,  0.,
                 -1.,  0.,
                  0.,  1.,
                  0., -1.;
            
            H3 << 15.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                  15.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                  15.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                  15.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                   0., 15.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                   0., 15.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                   0., 15.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                   0., 15.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                   0.,  0., 15.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                   0.,  0., 15.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                   0.,  0., 15.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                   0.,  0., 15.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                   0.,  0.,  0., 15.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                   0.,  0.,  0., 15.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                   0.,  0.,  0., 15.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                   0.,  0.,  0., 15.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                   0.,  0.,  0.,  0., 15.,  0.,  0.,  0.,  0.,  0.,  0.,
                   0.,  0.,  0.,  0., 15.,  0.,  0.,  0.,  0.,  0.,  0.,
                   0.,  0.,  0.,  0., 15.,  0.,  0.,  0.,  0.,  0.,  0.,
                   0.,  0.,  0.,  0., 15.,  0.,  0.,  0.,  0.,  0.,  0.,
                   0.,  0.,  0.,  0.,  0., 15.,  0.,  0.,  0.,  0.,  0.,
                   0.,  0.,  0.,  0.,  0., 15.,  0.,  0.,  0.,  0.,  0.,
                   0.,  0.,  0.,  0.,  0., 15.,  0.,  0.,  0.,  0.,  0.,
                   0.,  0.,  0.,  0.,  0., 15.,  0.,  0.,  0.,  0.,  0.,
                   0.,  0.,  0.,  0.,  0.,  0., 15.,  0.,  0.,  0.,  0.,
                   0.,  0.,  0.,  0.,  0.,  0., 15.,  0.,  0.,  0.,  0.,
                   0.,  0.,  0.,  0.,  0.,  0., 15.,  0.,  0.,  0.,  0.,
                   0.,  0.,  0.,  0.,  0.,  0., 15.,  0.,  0.,  0.,  0.,
                   0.,  0.,  0.,  0.,  0.,  0.,  0., 15.,  0.,  0.,  0.,
                   0.,  0.,  0.,  0.,  0.,  0.,  0., 15.,  0.,  0.,  0.,
                   0.,  0.,  0.,  0.,  0.,  0.,  0., 15.,  0.,  0.,  0.,
                   0.,  0.,  0.,  0.,  0.,  0.,  0., 15.,  0.,  0.,  0.,
                   0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 15.,  0.,  0.,
                   0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 15.,  0.,  0.,
                   0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 15.,  0.,  0.,
                   0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 15.,  0.,  0.,
                   0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 15.,  0.,
                   0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 15.,  0.,
                   0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 15.,  0.,
                   0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 15.,  0.,
                   0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 15.,
                   0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 15.,
                   0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 15.,
                   0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 15.,
                   0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                   0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                   0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                   0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                   0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,
                   0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,
                   0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,
                   0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,
                   0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                   0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                   0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                   0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                   0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                   0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                   0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                   0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.;
                    
            x_goal << 8.5, 4.5, 0., 0.;
            
            int i_x, i_u, i_z, i_c, i_n, ss;
            string s;

            // This initialization is very necessary!
            for (int i_n=0; i_n<N; i_n++){
                for (int ii=0; ii<K_feas/N; ii++){
                    dual_feas_z[i_n].push_back(Eigen::MatrixXd::Zero(N*nz, 1));
                    dual_feas_param[i_n].push_back(Eigen::MatrixXd::Zero(dual_len, 1));
                    Sp[i_n].push_back(0.0);}
                feas_begin[i_n]=0;}

            for (int ii=0; ii<K_opt; ii++){
                dual_opt_z.push_back(Eigen::MatrixXd::Zero(N*nz, 1));
                dual_opt_param.push_back(Eigen::MatrixXd::Zero(dual_len, 1));
                dual_opt_const.push_back(0.0);
                Sq.push_back(0.0);
                opt_begin=0;}

            // Very useful: https://support.gurobi.com/hc/en-us/community/posts/360048134231-gurobi-enviroment-model-and-C-header-file-issue
            model_sub = new GRBModel(env);
            model_sub->set(GRB_DoubleParam_BarConvTol, 1e-4);
            model_sub->set(GRB_IntParam_OutputFlag, 0);
            
            for (i_x=0; i_x<nx; i_x++){s = "x0_" + to_string(i_x);             x0_sub[i_x] = model_sub->addVar(-inf, inf, 0.0, GRB_CONTINUOUS, s);}
            for (i_c=0; i_c<nc; i_c++){s = "h_theta_sub_" + to_string(i_c);    h_theta_sub[i_c] = model_sub->addVar(-inf, inf, 0.0, GRB_CONTINUOUS, s);}
            for (i_x=0; i_x<nx; i_x++){s = "h_d_theta_sub_" + to_string(i_x);  h_d_theta_sub[i_x] = model_sub->addVar(-inf, inf, 0.0, GRB_CONTINUOUS, s);}

            for (i_n=0; i_n<(N+1); i_n++){
                for (i_x=0; i_x<nx; i_x++){s = "x_sub_t_" + to_string(i_n) + "_item_" + to_string(i_x);   
                                           x_sub[i_n][i_x] = model_sub->addVar(-inf, inf, 0.0, GRB_CONTINUOUS, s);}}

            for (i_n=0; i_n<N; i_n++){
                for (i_u=0; i_u<nu; i_u++){s = "u_sub_t_" + to_string(i_n) + "_item_" + to_string(i_u);   
                                           u_sub[i_n][i_u] = model_sub->addVar(-inf, inf, 0.0, GRB_CONTINUOUS, s);}}

            for (i_n=0; i_n<N; i_n++){
                for (i_z=0; i_z<nz; i_z++){s = "z_sub_t_" + to_string(i_n) + "_item_" + to_string(i_z);   
                                           z_sub[i_n][i_z] = model_sub->addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS, s);}}

            // Initial conditions
            for (i_x=0; i_x<nx; i_x++){
                s = "x0_item_" + to_string(i_x);
                model_sub->addConstr(x_sub[0][i_x] == x0_sub[i_x], s);}

            // Markov dynamics
            for (i_n=0; i_n<N; i_n++){
                for (i_x=0; i_x<nx; i_x++){
                    s = "x_dyn_" + to_string(i_n) + "_item_" + to_string(i_x);
                    GRBLinExpr expr = h_d_theta_sub[i_x];
                    for (ss=0; ss<nx; ss++){expr += E(i_x, ss)*x_sub[i_n][ss];}
                    for (ss=0; ss<nu; ss++){expr += F(i_x, ss)*u_sub[i_n][ss];}
                    for (ss=0; ss<nz; ss++){expr += G(i_x, ss)*z_sub[i_n][ss];}
                    model_sub->addConstr(x_sub[i_n+1][i_x] == expr, s);}}

            // Control constraint
            for (i_n=0; i_n<N; i_n++){
                for (i_c=0; i_c<nc; i_c++){
                    s = "xuz_" + to_string(i_n) + "_item_" + to_string(i_c);
                    GRBLinExpr expr = 0;
                    for (ss=0; ss<nx; ss++){expr += H1(i_c, ss)*x_sub[i_n][ss];}
                    for (ss=0; ss<nu; ss++){expr += H2(i_c, ss)*u_sub[i_n][ss];}
                    for (ss=0; ss<nz; ss++){expr += H3(i_c, ss)*z_sub[i_n][ss];}
                    model_sub->addConstr(expr <= h_theta_sub[i_c], s);}}

            // Set h_d_theta_sub=0
            for (int i_x=0; i_x<nx; i_x++){
                h_d_theta_sub[i_x].set(GRB_DoubleAttr_LB, 0.0); h_d_theta_sub[i_x].set(GRB_DoubleAttr_UB, 0.0);}

            // Objective function   
            for (i_n=0; i_n<N; i_n++){
                for (i_u=0; i_u<nu; i_u++){
                    obj_sub += u_sub[i_n][i_u]*R(i_u, i_u)*u_sub[i_n][i_u];}}

            for (i_n=0; i_n<N+1; i_n++){
                for (i_x=0; i_x<nx; i_x++){
                    obj_sub += (x_sub[i_n][i_x]-x_goal[i_x])*Q(i_x, i_x)*(x_sub[i_n][i_x]-x_goal[i_x]);}}

            // for (i_x=0; i_x<nx; i_x++){
            //     obj_sub += (x_sub[N][i_x]-x_goal[i_x])*Qn(i_x, i_x)*(x_sub[N][i_x]-x_goal[i_x]);}

            model_sub->setObjective(obj_sub, GRB_MINIMIZE);

            // ==============================================================================================================================
            model_infeas = new GRBModel(env);
            model_infeas->set(GRB_IntParam_InfUnbdInfo, 1);
            model_infeas->set(GRB_IntParam_OutputFlag, 0);

            for (i_x=0; i_x<nx; i_x++){s = "x0_" + to_string(i_x);                x0_infeas[i_x] = model_infeas->addVar(-inf, inf, 0.0, GRB_CONTINUOUS, s);}
            for (i_c=0; i_c<nc; i_c++){s = "h_theta_infeas_" + to_string(i_c);    h_theta_infeas[i_c] = model_infeas->addVar(-inf, inf, 0.0, GRB_CONTINUOUS, s);}
            for (i_x=0; i_x<nx; i_x++){s = "h_d_theta_infeas_" + to_string(i_x);  h_d_theta_infeas[i_x] = model_infeas->addVar(-inf, inf, 0.0, GRB_CONTINUOUS, s);}

            for (i_n=0; i_n<(N+1); i_n++){
                for (i_x=0; i_x<nx; i_x++){s = "x_infeas_t_" + to_string(i_n) + "_item_" + to_string(i_x);   
                                           x_infeas[i_n][i_x] = model_infeas->addVar(-inf, inf, 0.0, GRB_CONTINUOUS, s);}}

            for (i_n=0; i_n<N; i_n++){
                for (i_u=0; i_u<nu; i_u++){s = "u_infeas_t_" + to_string(i_n) + "_item_" + to_string(i_u);   
                                           u_infeas[i_n][i_u] = model_infeas->addVar(-inf, inf, 0.0, GRB_CONTINUOUS, s);}}

            for (i_n=0; i_n<N; i_n++){
                for (i_z=0; i_z<nz; i_z++){s = "z_infeas_t_" + to_string(i_n) + "_item_" + to_string(i_z);   
                                           z_infeas[i_n][i_z] = model_infeas->addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS, s);}}

            // Initial conditions
            for (i_x=0; i_x<nx; i_x++){
                s = "x0_item_" + to_string(i_x);
                model_infeas->addConstr(x_infeas[0][i_x] == x0_infeas[i_x], s);}

            // Markov dynamics
            for (i_n=0; i_n<N; i_n++){
                for (i_x=0; i_x<nx; i_x++){
                    s = "x_dyn_" + to_string(i_n) + "_item_" + to_string(i_x);
                    GRBLinExpr expr = h_d_theta_infeas[i_x];
                    for (ss=0; ss<nx; ss++){expr += E(i_x, ss)*x_infeas[i_n][ss];}
                    for (ss=0; ss<nu; ss++){expr += F(i_x, ss)*u_infeas[i_n][ss];}
                    for (ss=0; ss<nz; ss++){expr += G(i_x, ss)*z_infeas[i_n][ss];}
                    model_infeas->addConstr(x_infeas[i_n+1][i_x] == expr, s);}}

            // Control constraint
            for (i_n=0; i_n<N; i_n++){
                for (i_c=0; i_c<nc; i_c++){
                    s = "xuz_" + to_string(i_n) + "_item_" + to_string(i_c);
                    GRBLinExpr expr = 0;
                    for (ss=0; ss<nx; ss++){expr += H1(i_c, ss)*x_infeas[i_n][ss];}
                    for (ss=0; ss<nu; ss++){expr += H2(i_c, ss)*u_infeas[i_n][ss];}
                    for (ss=0; ss<nz; ss++){expr += H3(i_c, ss)*z_infeas[i_n][ss];}
                    model_infeas->addConstr(expr <= h_theta_infeas[i_c], s);}}

            // Set h_d_theta_infeas=0
            for (int i_x=0; i_x<nx; i_x++){
                h_d_theta_infeas[i_x].set(GRB_DoubleAttr_LB, 0.0); h_d_theta_infeas[i_x].set(GRB_DoubleAttr_UB, 0.0);}

            model_infeas->setObjective(obj_infeas, GRB_MINIMIZE);
        }

        ~MLD_GBD(){
            Q.setZero(); R.setZero(); Qn.setZero(); E.setZero(); F.setZero(); G.setZero(); H1.setZero(); H2.setZero(); H3.setZero(); x_goal.setZero();}

        void update_x0(const Vectord<nx>& x0_new, const Vectord<nc>& h_theta_new){

            // Generate in_param
            for (int i_x=0; i_x<nx; i_x++){
                in_param(i_x) = x0_new(i_x);}

            for (int i_n=0; i_n<N; i_n++){
                for (int i_c=0; i_c<nc; i_c++){
                    in_param((N+1)*nx+i_n*nc+i_c) = h_theta_new(i_c);}
            }
            
            // Update initial conditions for subproblems
            for (int i_x=0; i_x<nx; i_x++){
                x0_sub[i_x].set(GRB_DoubleAttr_LB, x0_new(i_x));    x0_sub[i_x].set(GRB_DoubleAttr_UB, x0_new(i_x));
                x0_infeas[i_x].set(GRB_DoubleAttr_LB, x0_new(i_x)); x0_infeas[i_x].set(GRB_DoubleAttr_UB, x0_new(i_x));}
            for (int i_c=0; i_c<nc; i_c++){
                h_theta_sub[i_c].set(GRB_DoubleAttr_LB, h_theta_new(i_c));    h_theta_sub[i_c].set(GRB_DoubleAttr_UB, h_theta_new(i_c));
                h_theta_infeas[i_c].set(GRB_DoubleAttr_LB, h_theta_new(i_c)); h_theta_infeas[i_c].set(GRB_DoubleAttr_UB, h_theta_new(i_c));}
        }

        void store_feas_cut(){
            // Can change using this pointer, so no input
            for (size_t i_n=0; i_n<N; i_n++){
                for (size_t i_p=0; i_p<new_dual_feas_z[i_n].size(); ++i_p){
                    dual_feas_z[i_n][feas_begin[i_n]] = new_dual_feas_z[i_n][i_p]; 
                    dual_feas_param[i_n][feas_begin[i_n]] = new_dual_feas_param[i_n][i_p]; 
                    feas_begin[i_n]++; 
                    
                    if (!feas_full[i_n]){feas_len[i_n]++;}

                    if(feas_begin[i_n]==K_feas/N){
                        feas_begin[i_n]=0;
                        feas_full[i_n]=true;}}

                new_dual_feas_z[i_n].clear();
                new_dual_feas_param[i_n].clear();
                } 
        }

        void store_opt_cut(int num_useful_cuts){
            // Can change using this pointer, so no input
            for (int i_p=0; i_p<num_useful_cuts; ++i_p){
                dual_opt_z[opt_begin] = new_dual_opt_z[i_p];
                dual_opt_param[opt_begin] = new_dual_opt_param[i_p];
                dual_opt_const[opt_begin] = new_dual_opt_const[i_p];
                opt_begin++;

                if (!opt_full){opt_len++;}

                if (opt_begin==K_opt){
                    opt_begin=0;
                    opt_full=true;}}
            
            new_dual_opt_z.clear();
            new_dual_opt_param.clear();
            new_dual_opt_const.clear();
        }

        void generate_Sp_and_Sq(){
            int i_p;
            for (int i_n=0; i_n<N; i_n++){
                for (i_p=0; i_p<feas_len[i_n]; i_p++){
                    Sp[i_n][i_p] = -dual_feas_param[i_n][i_p].dot(in_param);
                }
            }

            for (i_p=0; i_p<opt_len; i_p++){
                Sq[i_p] = dual_opt_const[i_p] - dual_opt_param[i_p].dot(in_param);
            }
        }

        void GCS_cost_map(::Eigen::Ref<::Eigen::Matrix<double, nz, 1>> in_cost_sou, 
                          ::Eigen::Ref<::Eigen::Matrix<double, nz, 1>> in_cost_tar, 
                          py::array_t<double> in_cost_edg){
            
            // Convert to Tensor. Reference: https://github.com/pybind/pybind11/issues/1377#issuecomment-691813711

            // request a buffer descriptor from Python
            py::buffer_info buffer_info = in_cost_edg.request();

            // extract data an shape of input array
            double *data = static_cast<double *>(buffer_info.ptr);
            std::vector<ssize_t> shape = buffer_info.shape;

            // wrap ndarray in Eigen::Map:
            Eigen::TensorMap<Eigen::Tensor<double, 3, Eigen::RowMajor>> T_cost_edg(data, shape[0], shape[1], shape[2]);

            for (int iv=0; iv<nz; iv++){
                cost_sou[iv] = in_cost_sou(iv);
                cost_tar[iv] = in_cost_tar(iv);}

            for (int in=0; in<N-2; in++){
                for (int iv1=0; iv1<nz; iv1++){
                    for (int iv2=0; iv2<nz; iv2++){
                        cost_edg[iv1][iv2][in] = T_cost_edg(iv1, iv2, in);}}}
        }
        
        // ===========================================================================================================================================================
        // Master problem solver
        // template <int nx, int nu, int nz, int nc, int N, int K_feas, int K_opt, int k_ah>
        pair<list<int>, list<double>> mat_solve_small_scale_MIP(const int i_t, const int prev_node, const array<vector<double>, N>& Sp, const vector<double>& Sq, 
                                                                const array<vector<double>, N>& new_Sp, const vector<double>& new_Sq){
            int ii, jj, ss, i_c, ah, ac_ah, ahead=min(k_ah+1, N-i_t);
            size_t s_c;
            list<int> ret_z;
            list<double> f_obj;
            double dot_sol, this_cost, this_max_cost, future, max_future, accumulated_future;
            bool ff;
            array<vector<double>, k_ah> d_Sp;  // Make them private members then clear
            array<vector<double>, k_ah> d_new_Sp;

            for (ah=1; ah<ahead; ah++){
                // cout << "===================================================================" << endl;
                // cout << "ah is " << ah << endl;   
                // Subtract max future terms in lookahead Sp

                // cout << "Generating stored cuts " << endl;
                for (i_c=0; i_c<feas_len[i_t+ah]; i_c++){
                    accumulated_future=0.0;
                    for (ac_ah=0; ac_ah<ah; ac_ah++){
                        // cout << "ac is " << ac_ah << endl;
                        max_future=-pow(10, 10);
                        for (ii=0; ii<num_sol; ii++){
                            future=0.0;
                            for (ss=0; ss<nz; ss++){
                                if (arr_z[ii][ss]==1){
                                    future += dual_feas_z[i_t+ah][i_c]((i_t+ac_ah+1)*nz+ss);
                                }
                            }
                            if (future>max_future){max_future=future;}
                        }
                    }
                    accumulated_future += max_future;
                    d_Sp[ah-1].push_back(accumulated_future);
                }

                // cout << "Generating new cuts " << endl;
                for (s_c=0; s_c<new_Sp[i_t+ah].size(); s_c++){
                    accumulated_future=0.0;
                    for (ac_ah=0; ac_ah<ah; ac_ah++){
                        // cout << "ac is " << ac_ah << endl;
                        max_future=-pow(10, 10);
                        for (ii=0; ii<num_sol; ii++){
                            future=0.0;
                            for (ss=0; ss<nz; ss++){
                                if (arr_z[ii][ss]==1){
                                    future += new_dual_feas_z[i_t+ah][s_c]((i_t+ac_ah+1)*nz+ss);
                                }
                            }
                            if (future>max_future){max_future=future;}
                        }
                    }
                    accumulated_future += max_future;
                    d_new_Sp[ah-1].push_back(accumulated_future);
                }
            }

            // Does this enumerating all solutions?? Isn't this kind of approach stupid?
            for (ii=0; ii<num_sol; ii++){
                // cout << "Checking solution " << arr_z[ii][0] << arr_z[ii][1] << endl;
                ff=true;
                for (ah=0; ah<ahead; ah++){    
                    // cout << "ah is " << ah << endl;                
                    if (ff){
                        // Check stored cuts
                        // cout << "Checking stored cuts " << endl;
                        for (i_c=0; i_c<feas_len[i_t+ah]; i_c++){  // Use feas_len to avoid 0 planes
                            dot_sol=0;
                            for (ss=0; ss<nz; ss++){
                                if (arr_z[ii][ss]==1){
                                    dot_sol += dual_feas_z[i_t+ah][i_c](i_t*nz+ss);
                                }
                            }
                            if (ah>=1){
                                if (dot_sol < (Sp[i_t+ah][i_c]-d_Sp[ah-1][i_c])){ff=false; break;}
                            }
                            else{
                                if (dot_sol < (Sp[i_t+ah][i_c])){ff=false; break;}
                            }
                        }
                    }
                    if (ff){
                        // Check new cuts
                        // cout << "Checking new cuts " << endl;
                        for (s_c=0; s_c<new_dual_feas_z[i_t+ah].size(); s_c++){
                            dot_sol=0;
                            for (ss=0; ss<nz; ss++){
                                if (arr_z[ii][ss]==1){
                                    dot_sol += new_dual_feas_z[i_t+ah][s_c](i_t*nz+ss);
                                }
                            }
                            if (ah>=1){
                                if (dot_sol < (new_Sp[i_t+ah][s_c]-d_new_Sp[ah-1][s_c])){ff=false; break;}
                            }
                            else{
                                if (dot_sol < (new_Sp[i_t+ah][s_c])){ff=false; break;}
                            }
                        }
                    }
                }

                if (ff){
                    // cout << "this solution is feasible!" << endl;
                    // this_max_cost = -pow(10, 10);   // Only need to change tihs part!!!!!!

                    if (i_t == 0){
                        this_max_cost = cost_sou[ii];
                    } 
                    else if (i_t == N-1){
                        this_max_cost = cost_tar[ii];
                    }
                    else{
                        this_max_cost = cost_edg[prev_node][ii][i_t-1];
                    }    

                    if (!Sq.empty()){
                        // Compute cost for stored cuts
                        for (jj=0; jj<opt_len; jj++){
                            this_cost = Sq[jj];
                            for (ss=0; ss<nz; ss++){
                                if (arr_z[ii][ss]==1){this_cost -= dual_opt_z[jj](i_t*nz+ss);}
                                }
                            if (this_cost>this_max_cost){this_max_cost = this_cost;}
                        }
                        // Compute cost for new cuts
                        for (s_c=0; s_c<new_Sq.size(); s_c++){
                            this_cost = new_Sq[s_c];
                            for (ss=0; ss<nz; ss++){
                                if (arr_z[ii][ss]==1){this_cost -= new_dual_opt_z[s_c](i_t*nz+ss);}
                                }
                            if (this_cost>this_max_cost){this_max_cost = this_cost;}
                        }
                    }
                    // Store solutions
                    auto pt = f_obj.begin(); auto ptf = ret_z.begin();
                    while((this_max_cost < *pt)&&(pt!=f_obj.end())){pt++; ptf++;}
                    f_obj.insert(pt, this_max_cost);
                    ret_z.insert(ptf, ii);
                    // insert_into_cost(ret_z, ret_z.begin(), next(ret_z.end(), -1), 
                    //                     f_obj, f_obj.begin(), next(f_obj.end(), -1), ret_z.size(), ii, this_max_cost);
                }
            }

            return make_pair(ret_z, f_obj);
        }

        // template <int nx, int nu, int nz, int nc, int N, int K_feas, int k_ah, int K_opt>
        pair<vector<int>, double> GBD_solve_master(){

            int i_t=0, ss, jj, i_n, prev_node;
            size_t kk;
            bool found_z; 
            vector<int> ls_z_opt;  
            // vector<vector<double>> ret_z_opt;  // For now put everything as double. But you may want to do some cast into int!!
            vector<list<int>> ls_all_z;
            list<int> ls_z_star;
            pair<list<int>, list<double>> small_mip_sol;

            array<vector<double>, N> in_Sp;
            array<vector<double>, N> new_in_Sp;
            vector<double> in_Sq;
            vector<double> new_in_Sq;

            // Make explicit copy
            for (i_n=0; i_n<N; i_n++){
                for (double ele:Sp[i_n]){in_Sp[i_n].push_back(ele);}}
            for (i_n=0; i_n<N; i_n++){
                for (double ele:new_Sp[i_n]){new_in_Sp[i_n].push_back(ele);}}
            for (double ele:Sq){in_Sq.push_back(ele);}
            for (double ele:new_Sq){new_in_Sq.push_back(ele);}

            while (i_t <= N-1){

                if (i_t == 0){prev_node = -1;}
                else{prev_node = ls_z_opt.back();}

                // Even during rewinding, this is solved multiple times??
                small_mip_sol = mat_solve_small_scale_MIP(i_t, prev_node, in_Sp, in_Sq, new_in_Sp, new_in_Sq);
                
                ls_z_star = get<0>(small_mip_sol);

                // cout << "Solutions are ";
                // for (auto ele:ls_z_star){cout << ele << ' ';}
                // cout << endl;

                if(ls_z_star.size()==0){
                    // Rewind to i_t where a feasible solution can be poped
                    found_z = false;
                    while(!found_z){
                        // cout << "Backtracking! " << endl;
                        i_t -= 1;
                        if (i_t < 0){throw invalid_argument("Master problem infeasible!!");}

                        // Undo the rhs changes using ls_z_opt[i_t] - what if I just save this somewhere before it was done? Depends on if copy is faster or algebra is faster.
                        for (ss=0; ss<nz; ss++){
                            if (arr_z[ls_z_opt[i_t]][ss]==1){
                                for (i_n=0; i_n<N; i_n++){  //i_n can in fact start from i_t
                                    for (jj=0; jj<feas_len[i_n]; jj++){
                                        in_Sp[i_n][jj] += dual_feas_z[i_n][jj](i_t*nz+ss);  // Redo one complete column
                                    }
                                    for (kk=0; kk<new_dual_feas_z[i_n].size(); kk++){
                                        new_in_Sp[i_n][kk] += new_dual_feas_z[i_n][kk](i_t*nz+ss);  // Redo one complete column
                                    }
                                }
                                for (jj=0; jj<opt_len; jj++){
                                    in_Sq[jj] += dual_opt_z[jj](i_t*nz+ss);  // Redo one complete column
                                }
                                for (kk=0; kk<new_dual_opt_z.size(); kk++){
                                    new_in_Sq[kk] += new_dual_opt_z[kk](i_t*nz+ss);  // Redo one complete column
                                }
                            }
                        }

                        if (ls_all_z[i_t].empty()){ls_z_opt.pop_back(); ls_all_z.pop_back();}  // If empty, go back further
                        else{
                            ls_z_opt[i_t] = ls_all_z[i_t].back(); ls_all_z[i_t].pop_back(); // Get another solution //assert (ls_z_opt.size()==(i_t+1));
                            found_z = true;
                            // cout << "Another solution found at step " << i_t << " which is " << ls_z_opt[i_t] << endl;

                            // cout << "All recorded solutions are ..." << endl;
                            // for (int ii=0; ii<ls_all_z.size(); ii++){
                            //     cout << "Step " << ii << endl;
                            //     for (auto ele:ls_all_z[ii]){
                            //         cout << ele << ' ';
                            //     }
                            //     cout << endl;
                            // }

                            // cout << "Best solutions are ..." << endl;
                            // for (auto ele:ls_z_opt){
                            //     cout << ele << ' ';
                            // }
                            // cout << endl;
                        }
                    }  
                }
                else{

                    // After backtracking, push_back is no longer the correct position to insert!
                    ls_all_z.push_back(ls_z_star);
                    ls_z_opt.push_back(ls_all_z[i_t].back()); 
                    
                    // cout << "All recorded solutions are ..." << endl;
                    // for (int ii=0; ii<ls_all_z.size(); ii++){
                    //     cout << "Step " << ii << endl;
                    //     for (auto ele:ls_all_z[ii]){
                    //         cout << ele << ' ';
                    //     }
                    //     cout << endl;
                    // }

                    // cout << "Best solutions are ..." << endl;
                    // for (auto ele:ls_z_opt){
                    //     cout << ele << ' ';
                    // }
                    // cout << endl;
                    
                    ls_all_z[i_t].pop_back();  //assert (ls_z_opt.size()==(i_t+1));

                    // If the last time step, return f_obj.
                    if (i_t == N-1){
                        // for (auto ele:ls_z_opt){ret_z_opt.push_back(arr_z[ele]);}
                        return make_pair(ls_z_opt, get<1>(small_mip_sol).back());}
                }

                // cout << "==============================================================" << endl;

                for (ss=0; ss<nz; ss++){
                    if (arr_z[ls_z_opt[i_t]][ss]==1){
                        for (i_n=0; i_n<N; i_n++){  //i_n can in fact start from i_t
                            for (jj=0; jj<feas_len[i_n]; jj++){
                                in_Sp[i_n][jj] -= dual_feas_z[i_n][jj](i_t*nz+ss);  // Redo one complete column
                            }
                            for (kk=0; kk<new_dual_feas_z[i_n].size(); kk++){
                                new_in_Sp[i_n][kk] -= new_dual_feas_z[i_n][kk](i_t*nz+ss);  // Redo one complete column
                            }
                        }
                        for (jj=0; jj<opt_len; jj++){
                            in_Sq[jj] -= dual_opt_z[jj](i_t*nz+ss);  // Redo one complete column
                        }
                        for (kk=0; kk<new_dual_opt_z.size(); kk++){
                            new_in_Sq[kk] -= new_dual_opt_z[kk](i_t*nz+ss);  // Redo one complete column
                        }
                    }
                }

                i_t += 1;
            } 

            // for (auto ele:ls_z_opt){ret_z_opt.push_back(arr_z[ele]);}
            return make_pair(ls_z_opt, -pow(2,10));
            
        }

        bool optimize_subproblem(int z_input[N][nz], double x_sol[N+1][nx], double u_sol[N][nu], double& f_obj, 
                                 stack<Vectord<N*nz>>& dual_z, stack<Vectord<dual_len>>& dual_param, double& const_part){

            int i_x, i_u, i_z, i_c, i_n, idd;
            double duals[dual_len] = {0.0};
            Vectord<N*nz> this_dual_z; this_dual_z.setZero();
            Vectord<dual_len> this_dual_param; this_dual_param.setZero();
            double tmp;
            bool all_zero=true;

            // Update z in subproblem
            for (i_n=0; i_n<N; i_n++){
                for (i_z=0; i_z<nz; i_z++){
                    z_sub[i_n][i_z].set(GRB_DoubleAttr_LB, z_input[i_n][i_z]); z_sub[i_n][i_z].set(GRB_DoubleAttr_UB, z_input[i_n][i_z]);}}

            model_sub->optimize();

            optimstatus = model_sub->get(GRB_IntAttr_Status);

            if (optimstatus == GRB_OPTIMAL){
                // Get solution
                for (i_n=0; i_n<(N+1); i_n++){
                    for (i_x=0; i_x<nx; i_x++){
                        x_sol[i_n][i_x] = x_sub[i_n][i_x].get(GRB_DoubleAttr_X);}}

                for (i_n=0; i_n<N; i_n++){
                    for (i_u=0; i_u<nu; i_u++){
                        u_sol[i_n][i_u] = u_sub[i_n][i_u].get(GRB_DoubleAttr_X);}}

                // // Get objective
                f_obj=0.0;
            
                for (i_n=0; i_n<N; i_n++){
                    for (i_u=0; i_u<nu; i_u++){
                        f_obj += u_sol[i_n][i_u]*R(i_u, i_u)*u_sol[i_n][i_u];}}

                for (i_n=0; i_n<N+1; i_n++){
                    for (i_x=0; i_x<nx; i_x++){
                        f_obj += (x_sol[i_n][i_x]-x_goal[i_x])*Q(i_x, i_x)*(x_sol[i_n][i_x]-x_goal[i_x]);}}

                // for (i_x=0; i_x<nx; i_x++){
                //     f_obj += (x_sol[N][i_x]-x_goal[i_x])*Qn(i_x, i_x)*(x_sol[N][i_x]-x_goal[i_x]);}

                // f_obj = model_sub->get(GRB_DoubleAttr_ObjVal);  // Once a while getting negative cost, why ??

                // Get dual, or do this: https://support.gurobi.com/hc/en-us/community/posts/4404830880145-Referencing-constraints-and-retrieving-their-dual-Pi-values-with-C-
                for (idd=0; idd<dual_len; idd++){
                    duals[idd] = -model_sub->getConstrByName(IDs.dual_name[idd]).get(GRB_DoubleAttr_Pi);}
                
                // Compute dual_z
                for (i_n=0; i_n<N; i_n++){
                    for (i_z=0; i_z<nz; i_z++){
                        tmp=0;
                        for (i_x=0; i_x<nx; i_x++){
                            tmp += duals[IDs.id_x_dyn[i_n][i_x]]*G(i_x, i_z);}
                        for (i_c=0; i_c<nc; i_c++){
                            tmp -= duals[IDs.id_xuz[i_n][i_c]]*H3(i_c, i_z);} 
                        this_dual_z(i_n*nz+i_z) = tmp;}}
                dual_z.push(this_dual_z);

                // Compute dual_param
                for (i_x=0; i_x<nx; i_x++){
                    this_dual_param(i_x) = duals[IDs.id_x0[i_x]];}
                for (i_n=0; i_n<N; i_n++){
                    for (i_x=0; i_x<nx; i_x++){
                        this_dual_param((i_n+1)*nx+i_x) = duals[IDs.id_x_dyn[i_n][i_x]];}}
                for (i_n=0; i_n<N; i_n++){
                    for (i_c=0; i_c<nc; i_c++){
                        this_dual_param((N+1)*nx+i_n*nc+i_c) = duals[IDs.id_xuz[i_n][i_c]];}}
                dual_param.push(this_dual_param);

                // Compute const_part
                const_part = f_obj;
                for (i_x=0; i_x<nx; i_x++){
                    const_part += x0_sub[i_x].get(GRB_DoubleAttr_X)*duals[IDs.id_x0[i_x]];}
                for (i_n=0; i_n<N; i_n++){
                    for (i_c=0; i_c<nc; i_c++){
                        const_part += h_theta_sub[i_c].get(GRB_DoubleAttr_X)*duals[IDs.id_xuz[i_n][i_c]];}}
                for (i_n=0; i_n<N; i_n++){
                    for (i_z=0; i_z<nz; i_z++){
                        for (i_x=0; i_x<nx; i_x++){const_part += duals[IDs.id_x_dyn[i_n][i_x]]*G(i_x, i_z)*z_sub[i_n][i_z].get(GRB_DoubleAttr_X);}
                        for (i_c=0; i_c<nc; i_c++){const_part -= duals[IDs.id_xuz[i_n][i_c]]*H3(i_c, i_z)*z_sub[i_n][i_z].get(GRB_DoubleAttr_X);}}}

                return true;
            }
            else{
                // if (optimstatus == GRB_INF_OR_UNBD || optimstatus == GRB_INFEASIBLE || optimstatus == GRB_UNBOUNDED){
                //     cout << " The model cannot be solved " << " because it is infeasible or unbounded " << endl;}
                // if (optimstatus != GRB_OPTIMAL){cout << " Optimization was stopped with status " << optimstatus << endl;}

                // Update z in subproblem
                for (i_n=0; i_n<N; i_n++){
                    for (i_z=0; i_z<nz; i_z++){
                        z_infeas[i_n][i_z].set(GRB_DoubleAttr_LB, z_input[i_n][i_z]); z_infeas[i_n][i_z].set(GRB_DoubleAttr_UB, z_input[i_n][i_z]);}}

                model_infeas->optimize();

                // Get Farkas proof
                for (idd=0; idd<dual_len; idd++){
                    duals[idd] = model_infeas->getConstrByName(IDs.dual_name[idd]).get(GRB_DoubleAttr_FarkasDual);}

                // Compute dual_z
                for (i_n=0; i_n<N; i_n++){
                    for (i_z=0; i_z<nz; i_z++){
                        tmp=0;
                        for (i_x=0; i_x<nx; i_x++){
                            tmp += duals[IDs.id_x_dyn[i_n][i_x]]*G(i_x, i_z);}
                        for (i_c=0; i_c<nc; i_c++){
                            tmp -= duals[IDs.id_xuz[i_n][i_c]]*H3(i_c, i_z);}
                        this_dual_z(i_n*nz+i_z) = tmp;}}
                dual_z.push(this_dual_z);

                // Compute dual_param
                for (i_x=0; i_x<nx; i_x++){
                    this_dual_param(i_x) = duals[IDs.id_x0[i_x]];}
                for (i_n=0; i_n<N; i_n++){
                    for (i_x=0; i_x<nx; i_x++){
                        this_dual_param((i_n+1)*nx+i_x) = duals[IDs.id_x_dyn[i_n][i_x]];}}
                for (i_n=0; i_n<N; i_n++){
                    for (i_c=0; i_c<nc; i_c++){
                        this_dual_param((N+1)*nx+i_n*nc+i_c) = duals[IDs.id_xuz[i_n][i_c]];}}
                dual_param.push(this_dual_param);

                do{
                    all_zero = true;

                    this_dual_z.setZero();
                    for (i_n=0; i_n<N-1; i_n++){
                        for (i_z=0; i_z<nz; i_z++){
                            if (dual_z.top()((i_n+1)*nz+i_z) != 0){this_dual_z(i_n*nz+i_z) = dual_z.top()((i_n+1)*nz+i_z); all_zero = false;}
                        }
                    }       

                    this_dual_param.setZero();
                    for (i_x=0; i_x<nx; i_x++){
                        if (dual_param.top()[IDs.id_x_dyn[0][i_x]] != 0)
                        {this_dual_param(i_x) = dual_param.top()[IDs.id_x_dyn[0][i_x]]; all_zero = false;}}

                    for (i_n=0; i_n<N-1; i_n++){
                        for (i_x=0; i_x<nx; i_x++){
                            if (dual_param.top()[IDs.id_x_dyn[i_n+1][i_x]] != 0)
                            {this_dual_param((i_n+1)*nx+i_x) = dual_param.top()[IDs.id_x_dyn[i_n+1][i_x]]; all_zero = false;}}}

                    for (i_n=0; i_n<N-1; i_n++){
                        for (i_c=0; i_c<nc; i_c++){
                            if (dual_param.top()[IDs.id_xuz[i_n+1][i_c]] != 0)
                            {this_dual_param((N+1)*nx+i_n*nc+i_c) = dual_param.top()[IDs.id_xuz[i_n+1][i_c]]; all_zero = false;}}}

                    if (!all_zero){
                    dual_z.push(this_dual_z);
                    dual_param.push(this_dual_param);}

                }while(!all_zero);

                return false;
            }
        }

        map<string, double> main_loop(::Eigen::Ref<::Eigen::Matrix<double, 4, 1>> x0_new, 
                                      ::Eigen::Ref<::Eigen::Matrix<double, 60, 1>> h_theta_new){
            cout << "==============================================================================" << endl;
            // auto t1 = chrono::steady_clock::now();

            int max_Benders_loop=300; // TODO: Another parameter!!!
            double MIPGap=0.1; // TODO: Another parameter!!!

            vector<double> list_f_obj_LB;
            vector<double> list_f_obj_UB;
            vector<string> ls_feas;

            int z_input[N][nz] = {0};

            // Current solution
            double x_sol[N+1][nx]={0};
            double u_sol[N][nu]={0};
            double cost=0;
            double const_part=0;
            double this_Sq=0, this_Sp=0;      

            // Best solution
            double x_best[N+1][nx]; 
            double u_best[N][nu]; 
            double z_best[N][nz];
            double cost_best=pow(10, 10);
            int ct_loop=0;
            int num_opt_cut=0; 
            int num_feas_cut=0;   
            
            int i_n, i_x, i_z, i_loop;
            bool feas;
            double current_gap;

            stack<Vectord<N*nz>> dual_z; 
            stack<Vectord<dual_len>> dual_param;
            
            update_x0(x0_new, h_theta_new);

            // auto t1 = chrono::steady_clock::now();
            generate_Sp_and_Sq();
            // auto t2 = chrono::steady_clock::now();
            // chrono::duration<double> time_span = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
            // cout << "Pre_process took " << 1000*time_span.count() << " ms" << endl;

            for (i_loop=0; i_loop<max_Benders_loop; i_loop++){
                
                // cout << "Iteration " << i_loop << " ==============================================================================" << endl;

                // First, solve master problem to obtain a contact sequence

                // auto t3 = chrono::steady_clock::now();
                auto ret_master = GBD_solve_master();
                // auto t4 = chrono::steady_clock::now();
                // chrono::duration<double> time_span2 = chrono::duration_cast<chrono::duration<double>>(t4 - t3);
                // cout << "Master solve took " << 1000*time_span2.count() << " ms" << endl;

                // Record solution
                // cout << "Master problem generate solution..." << endl;
                auto zz = get<0>(ret_master);
                for (i_n=0; i_n<N; i_n++){
                    for (i_z=0; i_z<nz; i_z++){
                        // cout << arr_z[zz[i_n]][i_z] << ' ';
                        z_input[i_n][i_z] = arr_z[zz[i_n]][i_z];}
                    // cout << endl;
                }

                // cout << "With cost ... " << get<1>(ret_master) << endl;

                list_f_obj_LB.push_back(get<1>(ret_master));
                
                if (i_loop > 0){
                    // check to see if the bound is tight.
                    if (list_f_obj_LB.back() > list_f_obj_UB.back()){current_gap = 0.0;}   
                    else{current_gap = abs(list_f_obj_UB.back() - list_f_obj_LB.back())/abs(list_f_obj_UB.back());}
                    if (current_gap <= MIPGap){
                        
                        cout << "feasibility ..." << endl;
                        for (auto ele:ls_feas){cout << ele << ' ';}
                        cout << endl;

                        cout << "UB ..." << endl;
                        for (auto ele:list_f_obj_UB){cout << ele << ' ';}
                        cout << endl;

                        cout << "LB ..." << endl;
                        for (auto ele:list_f_obj_LB){cout << ele << ' ';}
                        cout << endl;

                        break;}}                

                // Store optimality cuts necessary for generating optimal solutions, not proving optimal solutions. 
                // Can do better here by removing all optimality cuts after the optimal solution appears.
                // However, feasibility cuts are always needed hence stored immediately.
                // if (!init && feas){self.store_a_optimality_cut(ret_dual, const_part)}  // Here, keep a trace of the time when opt_cuts need to be stored!

                //Then, solve subproblem to add cuts to the master problem
                // auto t5 = chrono::steady_clock::now();
                feas = optimize_subproblem(z_input, x_sol, u_sol, cost, dual_z, dual_param, const_part);
                // auto t6 = chrono::steady_clock::now();
                // chrono::duration<double> time_span3 = chrono::duration_cast<chrono::duration<double>>(t6 - t5);
                // cout << "Subprob solve took " << 1000*time_span3.count() << " ms" << endl;

                ct_loop += 1;

                // auto t7 = chrono::steady_clock::now();
                if (feas){
                    // cout << "Feasible!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1" << endl;
                    ls_feas.push_back("feas");

                    // If feasible, add an optimal cut (no matter better or not)
                    // However, be careful about the optimality cut - you want to keep track of when the optimal solution is achieved.
                    new_dual_opt_z.push_back(dual_z.top());
                    new_dual_opt_param.push_back(dual_param.top());
                    new_dual_opt_const.push_back(const_part);
                    this_Sq = const_part - dual_param.top().dot(in_param);  // Should use sparse, so this dot-product is fast.
                    new_Sq.push_back(this_Sq);
                    dual_z.pop(); dual_param.pop();

                    // If solution is better, update.
                    if (cost < cost_best){
                        cost_best = cost; 
                        copy(&x_sol[0][0], &x_sol[0][0]+(N+1)*nx, &x_best[0][0]);
                        copy(&u_sol[0][0], &u_sol[0][0]+N*nu, &u_best[0][0]);
                        copy(&z_input[0][0], &z_input[0][0]+N*nz, &z_best[0][0]);
                        list_f_obj_UB.push_back(cost); 
                    }
                    // If solution is worse, keep.
                    else{list_f_obj_UB.push_back(list_f_obj_UB.back());}
                }
                            
                else{
                    // cout << "Infeasible!!!!!!!!!" << endl;
                    ls_feas.push_back("infeas");

                    // If infeasible, add a feasible cut
                    i_n=0;
                    while (!dual_z.empty()){
                        new_dual_feas_z[i_n].push_back(dual_z.top());
                        new_dual_feas_param[i_n].push_back(dual_param.top());
                        this_Sp = -dual_param.top().dot(in_param);
                        new_Sp[i_n].push_back(this_Sp);
                        dual_z.pop(); dual_param.pop();
                        i_n++;
                    }

                    if (i_loop == 0){list_f_obj_UB.push_back(pow(10,10));}
                    else{list_f_obj_UB.push_back(list_f_obj_UB.back());}
                }

                // auto t8 = chrono::steady_clock::now();
                // chrono::duration<double> time_span4 = chrono::duration_cast<chrono::duration<double>>(t8 - t7);
                // cout << "Saving solution took " << 1000*time_span4.count() << " ms" << endl;

            }

            // TODO: you also need a "max loop return"!!

            // Number of cuts include the new cuts, indicating the acutal number "necessary"
            num_opt_cut = opt_len + new_Sq.size();
            num_feas_cut = accumulate(begin(feas_len), end(feas_len), 0);
            for (i_n=0; i_n<N; i_n++){num_feas_cut += new_Sp[i_n].size();}

            // Removing all optimality cuts after the optimal solution appears ==================================================
            i_loop=0;
            for (auto ele:list_f_obj_UB){
                if (ele >= cost_best){break;}
                if (ls_feas[i_loop]=="feas"){i_loop+=1;}}  
            // =================================================================================================================

            // It seems like this is more correct!
            i_loop = new_Sq.size();
            
            // auto t9 = chrono::steady_clock::now();
            store_opt_cut(i_loop);  // Keep a trace of up to what point the cuts need to be stored
            store_feas_cut();
            // The code above should clean out new_dual_opt_z, new_dual_opt_param, new_dual_opt_const, new_dual_feas_z, new_dual_feas_param.
            // Now clean out new_Sp, new_Sq
            for (i_n=0; i_n<N; i_n++){new_Sp[i_n].clear();}
            new_Sq.clear();
            // auto t10 = chrono::steady_clock::now();
            // chrono::duration<double> time_span5 = chrono::duration_cast<chrono::duration<double>>(t10 - t9);
            // cout << "Saving cuts took " << 1000*time_span5.count() << " ms" << endl;

            // auto t2 = chrono::steady_clock::now();
            // chrono::duration<double> time_span = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
            // cout << "Main loop took " << 1000*time_span.count() << " ms" << " or " << 1/time_span.count() << " Hz" << endl;

            map<string, double> ret;

            // Search for the first point that is further from x0 than 0.15
            for (i_n=1; i_n<N; i_n++){
                if (sqrt( pow(x_best[i_n][0]-x_best[0][0], 2) + pow(x_best[i_n][1]-x_best[0][1], 2) ) > 0.15){
                    ret["control_x"] = u_best[i_n-1][0];
                    ret["control_y"] = u_best[i_n-1][1];
                    cout << "Use control at time " << i_n-1 << " ###########################################" << endl;
                    break;
                }                
            }

            ret["cost"] = cost_best;
            ret["num_iter"] = ct_loop;
            ret["num_opt_cut"] = num_opt_cut;
            ret["num_feas_cut"] = num_feas_cut;

            // Returning x cost quite some time!!
            for (i_n=0; i_n<N+1; i_n++){
                for (i_x=0; i_x<nx; i_x++){
                    ret["x_" + to_string(i_n) + "_" + to_string(i_x)] = x_best[i_n][i_x];
                }
            }

            cout << "Opt cut: " << num_opt_cut << endl;
            cout << "Feas cut: " << num_feas_cut << endl;

            // for (i_n=0; i_n<N; i_n++){
            //     for (i_z=0; i_z<nz; i_z++){
            //         cout << z_best[i_n][i_z] << ' ';
            //     }
            //     cout << endl;
            // }
            // cout << cost_best << endl;

            return ret;
        }

    private:
        Matrixd<nx, nx> Q; Matrixd<nu, nu> R; Matrixd<nx, nx> Qn; Matrixd<nx, nx> E; Matrixd<nx, nu> F; Matrixd<nx, nz> G; 
        Matrixd<nc, nx> H1; Matrixd<nc, nu> H2; Matrixd<nc, nz> H3;  Vectord<nx> x_goal;

        // Parameters (e.g. initial conditins)
        // Tuning knobs

        // dual IDs
        id_dual<nx, nu, nz, nc, N> IDs;

        // Gurobi subproblem solvers
        GRBModel *model_sub;
        GRBVar x0_sub[nx];
        GRBVar h_theta_sub[nc];
        GRBVar h_d_theta_sub[nx];
        GRBVar x_sub[N+1][nx];
        GRBVar u_sub[N][nu];
        GRBVar z_sub[N][nz];
        GRBQuadExpr obj_sub=0.0;

        GRBModel *model_infeas;
        GRBVar x0_infeas[nx];
        GRBVar h_theta_infeas[nc];
        GRBVar h_d_theta_infeas[nx];
        GRBVar x_infeas[N+1][nx];
        GRBVar u_infeas[N][nu];
        GRBVar z_infeas[N][nz];
        GRBQuadExpr obj_infeas=0.0;

        // Parameter
        Vectord<dual_len> in_param = ::Eigen::MatrixXd::Zero(dual_len, 1);

        // Cutting plane members
        // Optimality cuts
        vector<Vectord<N*nz>> dual_opt_z;
        vector<Vectord<dual_len>> dual_opt_param;
        vector<double> dual_opt_const;

        vector<Vectord<N*nz>> new_dual_opt_z;
        vector<Vectord<dual_len>> new_dual_opt_param;
        vector<double> new_dual_opt_const;

        int opt_begin=0;
        int opt_len=0.0;
        bool opt_full=false;

        // Feasibility cuts
        array<vector<Vectord<N*nz>>, N> dual_feas_z;
        array<vector<Vectord<dual_len>>, N> dual_feas_param;

        array<vector<Vectord<N*nz>>, N> new_dual_feas_z;
        array<vector<Vectord<dual_len>>, N> new_dual_feas_param;

        int feas_begin[N]={0};
        array<int, N> feas_len={0};
        bool feas_full[N] = {false};

        // Sp and Sq
        array<vector<double>, N> Sp;
        vector<double> Sq;

        array<vector<double>, N> new_Sp;
        vector<double> new_Sq;

        // Cost from GCS
        double cost_sou[nz] = {0};
        double cost_tar[nz] = {0};
        double cost_edg[nz][nz][N-2] = {0};

        int optimstatus=0;
        double dot_th = 0.9;  // TODO: Another parameter here!!!
        double lambda_th = 2000;  // TODO: Another parameter here!!!
        double inf=pow(10, 10);
};

PYBIND11_MODULE(GBD_GCS_hybrid_N50, m){
    const int nx = 4;
    const int nu = 2;
    const int nz = 11;
    const int nc = 60;
    const int N = 50;
    const int K_feas = 4000;
    const int k_ah = 5;
    const int K_opt = 500;

    py::class_<MLD_GBD<nx, nu, nz, nc, N, K_feas, k_ah, K_opt>>(m, "GBD")
        .def(py::init())
        .def("main_loop", &MLD_GBD<nx, nu, nz, nc, N, K_feas, k_ah, K_opt>::main_loop)
        .def("GCS_cost_map", &MLD_GBD<nx, nu, nz, nc, N, K_feas, k_ah, K_opt>::GCS_cost_map);
}

// Changes: nz, and the arr_z, and G
// nc, two places, and H1, H2, H3
