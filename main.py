"""Master entry point for the AE 695 vehicle lateral dynamics project."""
import argparse
import logging
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from config import DEFAULT_PARAMS
from models.vehicle_parameters import parameter_dataframe
from models.pacejka_tire import pacejka_lateral_force
from models.nonlinear_bicycle import nonlinear_dynamics
from models.linearized_bicycle import (analytical_matrices, numerical_jacobians,
                                       symbolic_matrices)
from analysis.equilibrium import find_equilibrium
from analysis.open_loop import simulate_responses
from analysis.eigen_analysis import eigen_analysis
from analysis.transfer_functions import transfer_functions
from analysis.state_transition import step_response
from analysis.canonical_forms import (controllable_canonical,
                                      observable_canonical, jordan_real,
                                      complex_modal)
from analysis.controllability_observability import controllability, observability
from controllers.pole_placement import pole_placement_gain
from controllers.ackermann import ackermann_gain
from controllers.lyapunov_lqr import lqr_gain, reference_precompensator
from observers.full_order import (observer_pole_gain, observer_ackermann,
                                  observer_lqr)
from observers.reduced_order import reduced_order_gain
from simulations.controller_simulation import simulate_nonlinear_controller
from simulations.observer_simulation import (simulate_full_observer,
                                             simulate_reduced_observer)
from simulations.combined_simulation import simulate_combined
from visualization.plotting import savefig, set_labels
from visualization.tables import save_csv
from report.generate_report import generate_report


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
ROOT = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(ROOT, "results", "figures")
TABLE_DIR = os.path.join(ROOT, "results", "tables")
DATA_DIR = os.path.join(ROOT, "results", "data")


def ensure_dirs():
    for path in (FIG_DIR, TABLE_DIR, DATA_DIR):
        os.makedirs(path, exist_ok=True)


def save_matrix_csv(name, matrix, path):
    rows = []
    matrix = np.asarray(matrix)
    for i in range(matrix.shape[0]):
        row = {"matrix": name, "row": i}
        for j in range(matrix.shape[1]):
            row[f"col_{j}"] = matrix[i,j]
        rows.append(row)
    new = pd.DataFrame(rows)
    if os.path.exists(path):
        old = pd.read_csv(path)
        new = pd.concat([old, new], ignore_index=True)
    save_csv(new, path)


def run_complete(params=DEFAULT_PARAMS):
    ensure_dirs()
    params.validate()
    logging.info("Validating vehicle parameters and Pacejka model")
    save_csv(parameter_dataframe(params),
             os.path.join(TABLE_DIR, "vehicle_parameters.csv"))

    # Pacejka validation and plot
    alpha=np.linspace(-0.5,0.5,1001)
    fyf=pacejka_lateral_force(alpha,params.Bf,params.pacejka_C,params.Df,params.pacejka_E)
    fyr=pacejka_lateral_force(alpha,params.Br,params.pacejka_C,params.Dr,params.pacejka_E)
    gradient=params.pacejka_C*params.Bf*params.Df
    if abs(gradient-params.Cf)/params.Cf > 1e-9:
        raise RuntimeError("Front Pacejka stiffness does not match Cf")
    fig,ax=plt.subplots()
    ax.plot(np.rad2deg(alpha),fyf/1000,label="Front"); ax.plot(np.rad2deg(alpha),fyr/1000,label="Rear")
    set_labels(ax,"Tire slip angle (deg)","Lateral force (kN)",
               "Pacejka lateral tire force characteristics")
    ax.legend()
    savefig(fig,os.path.join(FIG_DIR,"01_pacejka_tire_forces.png"))

    equilibria=[find_equilibrium(params,0)]
    for deg in (0.5,1.0,2.0):
        try:
            equilibria.append(find_equilibrium(params,deg))
        except RuntimeError as exc:
            logging.warning(str(exc))
    equil=pd.DataFrame(equilibria)
    save_csv(equil,os.path.join(TABLE_DIR,"equilibrium_points.csv"))

    A,B,C,D,Cm,Dm=analytical_matrices(params)
    An,Bn=numerical_jacobians(params)
    symA,symB=symbolic_matrices(params)
    symAn=np.array(symA.tolist(),dtype=float).reshape(2,2)
    symBn=np.array(symB.tolist(),dtype=float).reshape(2,1)
    jac_table=pd.DataFrame([
        {"matrix":"A","max_abs_error_numerical":np.max(np.abs(An-A)),
         "max_abs_error_symbolic":np.max(np.abs(symAn-A))},
        {"matrix":"B","max_abs_error_numerical":np.max(np.abs(Bn-B)),
         "max_abs_error_symbolic":np.max(np.abs(symBn-B))}
    ])
    save_csv(jac_table,os.path.join(TABLE_DIR,"jacobian_validation.csv"))
    save_matrix_csv("A",A,os.path.join(TABLE_DIR,"linearized_matrices.csv"))
    save_matrix_csv("B",B,os.path.join(TABLE_DIR,"linearized_matrices.csv"))
    logging.info("A=%s",A); logging.info("B=%s",B)

    vals,vecs,eigen_rows,char_coeff=eigen_analysis(A)
    save_csv(pd.DataFrame(eigen_rows),os.path.join(TABLE_DIR,"eigenvalues.csv"))
    fig,ax=plt.subplots()
    ax.scatter(vals.real,vals.imag,s=60)
    set_labels(ax,"Real part (1/s)","Imaginary part (rad/s)","Open-loop eigenvalues")
    ax.axvline(0,color="k",lw=1)
    savefig(fig,os.path.join(FIG_DIR,"09_eigenvalues.png"))

    Mc,rankc,sc,Wc=controllability(A,B)
    Mo,ranko,so,Wo=observability(A,Cm)
    rankvy=np.linalg.matrix_rank(np.vstack([C[0:1],C[0:1]@A]))
    rankfull=np.linalg.matrix_rank(np.vstack([C,C@A]))
    system_properties_table=pd.DataFrame([
        {"analysis":"controllability","rank":rankc,"index":2,"singular_values":str(sc)},
        {"analysis":"yaw_observability","rank":ranko,"index":2,"singular_values":str(so)},
        {"analysis":"lateral_velocity_observability","rank":rankvy,"index":rankvy},
        {"analysis":"both_measurements_observability","rank":rankfull,"index":rankfull}
    ])
    save_csv(system_properties_table,os.path.join(TABLE_DIR,"system_properties.csv"))
    save_matrix_csv("Mc",Mc,os.path.join(TABLE_DIR,"controllability_matrix.csv"))
    save_matrix_csv("Mo",Mo,os.path.join(TABLE_DIR,"observability_matrix.csv"))

    numvy,numr,den=transfer_functions(A,B,C,D)
    nrows=max(len(numvy),len(numr),len(den))
    pd.DataFrame({"numerator_vy":list(numvy)+[np.nan]*(nrows-len(numvy)),
                  "numerator_r":list(numr)+[np.nan]*(nrows-len(numr)),
                  "denominator":list(den)+[np.nan]*(nrows-len(den))}).to_csv(
                      os.path.join(TABLE_DIR,"transfer_function_coefficients.csv"),index=False)

    cc=controllable_canonical(A,B,Cm)
    oc=observable_canonical(A,B,Cm)
    jvals,jT,jA=jordan_real(A)
    cvals,cT,cAj,cmodal_residual=complex_modal(A)
    canonical_rows=[
        {"form":"controllable","T_condition":cc["condition"],
         "residual":cc["residual"],"rank":cc["rank"]},
        {"form":"observable","T_condition":oc["condition"],
         "residual":oc["residual"],"rank":2},
        {"form":"complex_modal","T_condition":np.linalg.cond(cT),
         "residual":cmodal_residual,"rank":2},
        {"form":"real_modal","T_condition":np.linalg.cond(jT),
         "residual":np.linalg.norm(A@jT-jT@jA),"rank":2}
    ]
    save_csv(pd.DataFrame(canonical_rows),
             os.path.join(TABLE_DIR,"canonical_validation.csv"))
    for name,data in (("controllable",cc),("observable",oc)):
        save_matrix_csv(name+"_T",data["T"],os.path.join(TABLE_DIR,"canonical_matrices.csv"))
        save_matrix_csv(name+"_A",data["A"],os.path.join(TABLE_DIR,"canonical_matrices.csv"))
    save_matrix_csv("complex_modal_T",cT,os.path.join(TABLE_DIR,"canonical_matrices.csv"))
    save_matrix_csv("complex_modal_Aj",cAj,os.path.join(TABLE_DIR,"canonical_matrices.csv"))
    save_matrix_csv("real_modal_T",jT,os.path.join(TABLE_DIR,"canonical_matrices.csv"))
    save_matrix_csv("real_modal_Ar",jA,os.path.join(TABLE_DIR,"canonical_matrices.csv"))

    t=np.linspace(0,10,501)
    u05=lambda ti: np.deg2rad(0.5)
    nl_lin,nl_linear,extra=simulate_responses(params,A,B,t,u05,np.zeros(2))
    err=nl_lin-nl_linear
    rmse=np.sqrt(np.mean(err**2,axis=0))
    open_df=pd.DataFrame({"time_s":t,"vy_nonlinear":nl_lin[:,0],"r_nonlinear":nl_lin[:,1],
                          "vy_linear":nl_linear[:,0],"r_linear":nl_linear[:,1],
                          "vy_error":err[:,0],"r_error":err[:,1],
                          "alpha_f_deg":extra["alpha_f"],"alpha_r_deg":extra["alpha_r"],
                          "Fyf_N":extra["Fyf"],"Fyr_N":extra["Fyr"]})
    save_csv(open_df,os.path.join(DATA_DIR,"open_loop_0p5deg.csv"))
    fig,axes=plt.subplots(2,1,sharex=True)
    axes[0].plot(t,nl_lin[:,0],label="Nonlinear"); axes[0].plot(t,nl_linear[:,0],"--",label="Linear")
    set_labels(axes[0],"","Lateral velocity (m/s)")
    axes[1].plot(t,nl_lin[:,1],label="Nonlinear"); axes[1].plot(t,nl_linear[:,1],"--",label="Linear")
    set_labels(axes[1],"Time (s)","Yaw rate (rad/s)")
    axes[0].legend(); axes[1].legend()
    savefig(fig,os.path.join(FIG_DIR,"04_nonlinear_linear_open_loop.png"))
    open_loop_comparison=[]
    for steering_deg in (0.5,3.0,10.0):
        ufun=lambda ti,deg=steering_deg: np.deg2rad(deg)
        nonlinear,linear,extra_amp=simulate_responses(params,A,B,t,ufun,np.zeros(2))
        difference=nonlinear-linear
        rmse_amp=np.sqrt(np.mean(difference**2,axis=0))
        for state_index,state_name in enumerate(("vy","r")):
            open_loop_comparison.append({
                "steering_deg":steering_deg,"state":state_name,
                "rmse":rmse_amp[state_index],
                "max_abs_error":np.max(np.abs(difference[:,state_index]))
            })
        pd.DataFrame({"time_s":t,"vy_nonlinear":nonlinear[:,0],"r_nonlinear":nonlinear[:,1],
                      "vy_linear":linear[:,0],"r_linear":linear[:,1],
                      "vy_error":difference[:,0],"r_error":difference[:,1],
                      "alpha_f_deg":extra_amp["alpha_f"],"alpha_r_deg":extra_amp["alpha_r"],
                      "Fyf_N":extra_amp["Fyf"],"Fyr_N":extra_amp["Fyr"]}).to_csv(
                          os.path.join(DATA_DIR,f"open_loop_{steering_deg:g}deg.csv"),index=False)
    save_csv(pd.DataFrame(open_loop_comparison),
             os.path.join(TABLE_DIR,"open_loop_comparison.csv"))

    desired=np.array([-8,-10])
    K,Kpoles=pole_placement_gain(A,B,desired)
    Ka=ackermann_gain(A,B,desired)
    Kapoles=np.linalg.eigvals(A-B@Ka)
    K_lqr,P,ricc,lyap=lqr_gain(A,B)
    Klqrpoles=np.linalg.eigvals(A-B@K_lqr)
    C_yaw=np.array([[0.0,1.0]])
    Nbar=reference_precompensator(A,B,C_yaw,K,measured_yaw_only=True)
    Nbar_a=reference_precompensator(A,B,C_yaw,Ka,measured_yaw_only=True)
    Nbar_lqr=reference_precompensator(A,B,C_yaw,K_lqr,measured_yaw_only=True)

    desired_obs=np.array([-16,-18])
    L,opoles=observer_pole_gain(A,Cm,desired_obs)
    La,oapoles=observer_ackermann(A,Cm,desired_obs)
    Ll,olpoles,ores=observer_lqr(A,Cm)
    controller_table=pd.DataFrame([
        {"method":"pole_placement","gain_vy":L[0,0],"gain_r":L[1,0],"poles":str(opoles)},
        {"method":"dual_ackermann","gain_vy":La[0,0],"gain_r":La[1,0],"poles":str(oapoles)},
        {"method":"dual_lqr","gain_vy":Ll[0,0],"gain_r":Ll[1,0],"poles":str(olpoles)}
    ]).to_csv(os.path.join(TABLE_DIR,"observer_gains.csv"),index=False)
    Lr,reduced_pole=reduced_order_gain(A,B,-12)
    save_csv(pd.DataFrame([{"method":"reduced_order","gain":Lr,
                            "achieved_error_pole":reduced_pole,
                            "desired_error_pole":-12.0,
                            "dynamic_observer_states":1}]),
             os.path.join(TABLE_DIR,"reduced_observer_validation.csv"))

    def step_ref(ti): return np.deg2rad(2.0)
    def sine_ref(ti): return np.deg2rad(2.0*np.sin(0.5*ti))
    step_states,step_control=simulate_nonlinear_controller(params,A,B,K,Nbar,t,step_ref)
    step_states_a,step_control_a=simulate_nonlinear_controller(params,A,B,Ka,Nbar_a,t,step_ref)
    step_states_lqr,step_control_lqr=simulate_nonlinear_controller(params,A,B,K_lqr,Nbar_lqr,t,step_ref)
    sine_states,sine_control=simulate_nonlinear_controller(params,A,B,K,Nbar,t,sine_ref)
    sine_states_a,sine_control_a=simulate_nonlinear_controller(params,A,B,Ka,Nbar_a,t,sine_ref)
    sine_states_lqr,sine_control_lqr=simulate_nonlinear_controller(params,A,B,K_lqr,Nbar_lqr,t,sine_ref)
    step_error=step_ref(0)-step_states[:,1]
    sine_error=np.array([sine_ref(ti)-sine_states[k,1] for k,ti in enumerate(t)])
    step_rmse=np.sqrt(np.mean(step_error**2)); sine_rmse=np.sqrt(np.mean(sine_error**2))
    step_rmse_a=np.sqrt(np.mean((step_ref(0)-step_states_a[:,1])**2))
    step_rmse_lqr=np.sqrt(np.mean((step_ref(0)-step_states_lqr[:,1])**2))
    sine_rmse_a=np.sqrt(np.mean(np.array([sine_ref(ti)-sine_states_a[k,1]
                                          for k,ti in enumerate(t)])**2))
    sine_rmse_lqr=np.sqrt(np.mean(np.array([sine_ref(ti)-sine_states_lqr[k,1]
                                            for k,ti in enumerate(t)])**2))
    controller_table=pd.DataFrame([
        {"method":"pole_placement","gain_vy":K[0,0],"gain_r":K[0,1],"poles":str(Kpoles),
         "step_rmse_rad":step_rmse,"sine_rmse_rad":sine_rmse,
         "integral_control_effort":np.trapezoid(step_control**2,t),
         "max_steering_deg":np.max(np.abs(np.rad2deg(step_control))),
         "Nbar":Nbar},
        {"method":"ackermann","gain_vy":Ka[0,0],"gain_r":Ka[0,1],"poles":str(Kapoles),
         "step_rmse_rad":step_rmse_a,"sine_rmse_rad":sine_rmse_a,
         "integral_control_effort":np.trapezoid(step_control_a**2,t),
         "max_steering_deg":np.max(np.abs(np.rad2deg(step_control_a))),
         "Nbar":Nbar_a},
        {"method":"lqr","gain_vy":K_lqr[0,0],"gain_r":K_lqr[0,1],"poles":str(Klqrpoles),
         "step_rmse_rad":step_rmse_lqr,"sine_rmse_rad":sine_rmse_lqr,
         "integral_control_effort":np.trapezoid(step_control_lqr**2,t),
         "max_steering_deg":np.max(np.abs(np.rad2deg(step_control_lqr))),
         "Nbar":Nbar_lqr}
    ])
    save_csv(controller_table,os.path.join(TABLE_DIR,"controller_gains.csv"))
    pd.DataFrame({"time_s":t,"yaw_reference":np.array([np.rad2deg(step_ref(ti)) for ti in t]),
                  "yaw_actual":np.rad2deg(step_states[:,1]),
                  "vy":step_states[:,0],"steering_deg":np.rad2deg(step_control),
                  "tracking_error_rad":step_error}).to_csv(
                      os.path.join(DATA_DIR,"step_controller.csv"),index=False)

    fig,ax=plt.subplots()
    ax.plot(t,np.rad2deg([step_ref(ti) for ti in t]),label="Reference")
    ax.plot(t,np.rad2deg(step_states[:,1]),label="Actual")
    set_labels(ax,"Time (s)","Yaw rate (deg/s)","Nonlinear step yaw-rate tracking")
    ax.legend(); savefig(fig,os.path.join(FIG_DIR,"18_step_tracking.png"))
    fig,ax=plt.subplots()
    ax.plot(t,np.rad2deg([step_ref(ti) for ti in t])-np.rad2deg(step_states[:,1]))
    set_labels(ax,"Time (s)","Tracking error (deg/s)")
    savefig(fig,os.path.join(FIG_DIR,"22_tracking_error.png"))

    truth,xhat,measured=simulate_full_observer(params,A,B,L,Cm,u05,t,np.array([1.0,-0.1]),
                                                np.array([0.0,0.0]),0.0)
    truth_n,xhat_n,meas_n=simulate_full_observer(params,A,B,L,Cm,u05,t,np.array([1.0,-0.1]),
                                                  np.array([0.0,0.0]),0.05)
    observer_rmse=np.sqrt(np.mean((truth-xhat)**2,axis=0))
    noise_results=[]
    for sigma in (0.0,0.02,0.05,0.1):
        tr,xh,me=simulate_full_observer(params,A,B,L,Cm,u05,t,np.array([1.0,-0.1]),
                                         np.array([0.0,0.0]),sigma)
        noise_results.append({"observer":"full_order","sigma_rad_s":sigma,
                              "vy_rmse":np.sqrt(np.mean((tr[:,0]-xh[:,0])**2)),
                              "r_rmse":np.sqrt(np.mean((tr[:,1]-xh[:,1])**2)),
                              "vy_mae":np.mean(np.abs(tr[:,0]-xh[:,0]))})
        initial_vy=0.0
        initial_r=-0.1
        z_initial=initial_vy-Lr*initial_r
        trr,xhr,mer=simulate_reduced_observer(params,A,B,Lr,u05,t,
                                              np.array([1.0,initial_r]),
                                              z0=z_initial,noisy_sigma=sigma)
        noise_results.append({"observer":"reduced_order","sigma_rad_s":sigma,
                              "vy_rmse":np.sqrt(np.mean((trr[:,0]-xhr[:,0])**2)),
                              "r_rmse":np.sqrt(np.mean((trr[:,1]-xhr[:,1])**2)),
                              "vy_mae":np.mean(np.abs(trr[:,0]-xhr[:,0]))})
    save_csv(pd.DataFrame(noise_results),os.path.join(TABLE_DIR,"measurement_noise.csv"))

    noise_df=pd.DataFrame(noise_results)
    fig,axes=plt.subplots(2,1,sharex=True)
    for observer_name in ("full_order","reduced_order"):
        subset=noise_df[noise_df["observer"]==observer_name]
        axes[0].plot(subset["sigma_rad_s"],subset["vy_rmse"],marker="o",
                     label=observer_name.replace("_"," ").title())
        axes[1].plot(subset["sigma_rad_s"],subset["r_rmse"],marker="o",
                     label=observer_name.replace("_"," ").title())
    set_labels(axes[0],"","Lateral velocity RMSE (m/s)",
               "Observer noise sensitivity")
    set_labels(axes[1],"Measurement noise std. dev. (rad/s)",
               "Yaw-rate RMSE (rad/s)")
    axes[0].legend(); axes[1].legend()
    savefig(fig,os.path.join(FIG_DIR,"29_measurement_noise_rmse.png"))

    fig,axes=plt.subplots(2,1,sharex=True)
    axes[0].plot(t,truth[:,0],label="True"); axes[0].plot(t,xhat[:,0],"--",label="Estimate")
    set_labels(axes[0],"","Lateral velocity (m/s)")
    axes[1].plot(t,truth[:,1],label="True"); axes[1].plot(t,measured,label="Measured"); axes[1].plot(t,xhat[:,1],"--",label="Estimate")
    set_labels(axes[1],"Time (s)","Yaw rate (rad/s)")
    axes[0].legend(); axes[1].legend()
    savefig(fig,os.path.join(FIG_DIR,"23_full_observer_states.png"))

    truth_r,xhat_r,meas_r=simulate_reduced_observer(params,A,B,Lr,u05,t,
                                                     np.array([1.0,-0.1]),
                                                     z0=0.0-Lr*(-0.1),noisy_sigma=0.0)
    reduced_error=np.sqrt(np.mean((truth_r[:,0]-xhat_r[:,0])**2))
    combined_full=simulate_combined(params,A,B,K_lqr,Nbar_lqr,step_ref,t,"full",L=L)
    combined_red=simulate_combined(params,A,B,K_lqr,Nbar_lqr,step_ref,t,"reduced",Lr=Lr)

    combined_rows=[]
    combined_step_results={}
    combined_sine_results={}
    controller_configs=[
        ("pole_full","pole_placement","full",K,Nbar),
        ("pole_reduced","pole_placement","reduced",K,Nbar),
        ("ackermann_full","ackermann","full",Ka,Nbar_a),
        ("ackermann_reduced","ackermann","reduced",Ka,Nbar_a),
        ("lqr_full","lqr","full",K_lqr,Nbar_lqr),
        ("lqr_reduced","lqr","reduced",K_lqr,Nbar_lqr),
    ]
    for name,label,observer,kgain,ngain in controller_configs:
        step_data=simulate_combined(params,A,B,kgain,ngain,step_ref,t,
                                    observer=observer,L=L,Lr=Lr,
                                    x0=np.zeros(2),xhat0=np.zeros(2),
                                    return_diagnostics=True)
        sine_data=simulate_combined(params,A,B,kgain,ngain,sine_ref,t,
                                    observer=observer,L=L,Lr=Lr,
                                    x0=np.zeros(2),xhat0=np.zeros(2),
                                    return_diagnostics=True)
        for reference_name,reference_data in (("step",step_data),("sinusoid",sine_data)):
            states,estimates,controls,measured,references,errors=reference_data
            tracking_error=references-states[:,1]
            sat_mask=np.abs(np.abs(controls)-np.deg2rad(20))<1e-8
            row={"configuration":name,"reference":reference_name,
                 "controller":label,"observer":observer,
                 "tracking_rmse_rad":np.sqrt(np.mean(tracking_error**2)),
                 "vy_estimation_rmse_m_s":np.sqrt(np.mean(errors[:,0]**2)),
                 "r_estimation_rmse_rad_s":np.sqrt(np.mean(errors[:,1]**2)),
                 "max_estimation_error":np.max(np.linalg.norm(errors,axis=1)),
                 "max_steering_deg":np.max(np.abs(np.rad2deg(controls))),
                 "saturation_duration_s":np.sum(sat_mask)*np.mean(np.diff(t))}
            combined_rows.append(row)
        combined_step_results[name]=(step_data[0],step_data[1],step_data[2],
                                     step_data[3],step_data[4],step_data[5])
        combined_sine_results[name]=(sine_data[0],sine_data[1],sine_data[2],
                                     sine_data[3],sine_data[4],sine_data[5])
    save_csv(pd.DataFrame(combined_rows),
             os.path.join(TABLE_DIR,"controller_observer_comparison.csv"))
    fig,ax=plt.subplots()
    ax.plot(t,np.rad2deg([step_ref(ti) for ti in t]),"k--",label="Reference")
    for name in combined_step_results:
        ax.plot(t,np.rad2deg(combined_step_results[name][0][:,1]),label=name)
    set_labels(ax,"Time (s)","Yaw rate (deg/s)",
               "Combined controller-observer step tracking")
    ax.legend(fontsize=7,ncol=2)
    savefig(fig,os.path.join(FIG_DIR,"30_combined_step_comparison.png"))

    fig,ax=plt.subplots()
    ax.plot(t,np.rad2deg([sine_ref(ti) for ti in t]),"k--",label="Reference")
    for name in combined_sine_results:
        ax.plot(t,np.rad2deg(combined_sine_results[name][0][:,1]),label=name)
    set_labels(ax,"Time (s)","Yaw rate (deg/s)",
               "Combined controller-observer sinusoidal tracking")
    ax.legend(fontsize=7,ncol=2)
    savefig(fig,os.path.join(FIG_DIR,"31_combined_sine_comparison.png"))
    for name in ("lqr_full","lqr_reduced"):
        states,estimates,controls,measured,references,errors=combined_step_results[name]
        fig,axes=plt.subplots(3,1,sharex=True)
        axes[0].plot(t,states[:,0],label="Actual")
        axes[0].plot(t,estimates[:,0],"--",label="Estimated")
        set_labels(axes[0],"","Lateral velocity (m/s)")
        axes[1].plot(t,np.rad2deg(states[:,1]),label="Actual")
        axes[1].plot(t,np.rad2deg(estimates[:,1]),"--",label="Estimated")
        set_labels(axes[1],"","Yaw rate (deg/s)")
        axes[2].plot(t,np.rad2deg(controls))
        set_labels(axes[2],"Time (s)","Steering angle (deg)",
                   f"{name.replace('_',' ').title()} steering effort")
        axes[0].legend(); axes[1].legend()
        savefig(fig,os.path.join(FIG_DIR,
                                 f"32_{name}_states_errors_control.png"))
        fig,ax=plt.subplots()
        ax.plot(t,errors[:,0],label="Lateral velocity error")
        ax.plot(t,np.rad2deg(errors[:,1]),label="Yaw-rate error (deg/s)")
        set_labels(ax,"Time (s)","Estimation error",
                   f"{name.replace('_',' ').title()} observer errors")
        ax.legend()
        savefig(fig,os.path.join(FIG_DIR,
                                 f"33_{name}_estimation_errors.png"))

    # Speed and road-friction sensitivity studies.
    sensitivity_rows=[]
    for vx in (10.0,20.0,30.0):
        ps=__import__("dataclasses").replace(params,vx=vx)
        As,Bs,*_=analytical_matrices(ps)
        fixed_states,fixed_control=simulate_nonlinear_controller(
            ps,As,Bs,K_lqr,Nbar_lqr,t,step_ref)
        fixed_error=step_ref(0)-fixed_states[:,1]
        sensitivity_rows.append({"study":"speed","value":vx,
                                 "eigenvalues":str(np.linalg.eigvals(As)),
                                 "A11":As[0,0],"A12":As[0,1],
                                 "A21":As[1,0],"A22":As[1,1],
                                 "closed_loop_poles_fixed_gain":str(np.linalg.eigvals(As-Bs@K_lqr)),
                                 "Df":ps.Df,"Dr":ps.Dr,
                                 "step_rmse_rad":np.sqrt(np.mean(fixed_error**2)),
                                 "max_steering_deg":np.max(np.abs(np.rad2deg(fixed_control)))})
    for mu in (0.4,0.7,0.9):
        ps=__import__("dataclasses").replace(params,mu=mu)
        As,Bs,*_=analytical_matrices(ps)
        fixed_states,fixed_control=simulate_nonlinear_controller(
            ps,As,Bs,K_lqr,Nbar_lqr,t,step_ref)
        fixed_error=step_ref(0)-fixed_states[:,1]
        sensitivity_rows.append({"study":"friction","value":mu,
                                 "eigenvalues":str(np.linalg.eigvals(As)),
                                 "A11":As[0,0],"A12":As[0,1],
                                 "A21":As[1,0],"A22":As[1,1],
                                 "closed_loop_poles_fixed_gain":str(np.linalg.eigvals(As-Bs@K_lqr)),
                                 "Df":ps.Df,"Dr":ps.Dr,
                                 "step_rmse_rad":np.sqrt(np.mean(fixed_error**2)),
                                 "max_steering_deg":np.max(np.abs(np.rad2deg(fixed_control)))})
    save_csv(pd.DataFrame(sensitivity_rows),
             os.path.join(TABLE_DIR,"speed_friction_sensitivity.csv"))

    # State-transition samples and sine-reference tracking plot.
    zi,zs,total=step_response(A,B,np.zeros(2),np.deg2rad(2.0),t)
    phi_rows=[]
    for tt in t:
        phi=expm(A*tt)
        phi_rows.append({"time_s":tt,"phi11":phi[0,0],"phi12":phi[0,1],
                         "phi21":phi[1,0],"phi22":phi[1,1],
                         "zero_input_vy":zi[0,np.argmin(np.abs(t-tt))],
                         "zero_input_r":zi[1,np.argmin(np.abs(t-tt))],
                         "zero_state_vy":zs[0,np.argmin(np.abs(t-tt))],
                         "zero_state_r":zs[1,np.argmin(np.abs(t-tt))]})
    save_csv(pd.DataFrame(phi_rows),
             os.path.join(TABLE_DIR,"state_transition_samples.csv"))
    transition_df=pd.DataFrame({
        "time_s":t,
        "zero_input_vy_m_s":zi[0],
        "zero_input_r_rad_s":zi[1],
        "zero_state_vy_m_s":zs[0],
        "zero_state_r_rad_s":zs[1],
        "total_vy_m_s":total[0],
        "total_r_rad_s":total[1]
    })
    save_csv(transition_df,os.path.join(DATA_DIR,"state_transition_responses.csv"))
    fig,axes=plt.subplots(3,1,sharex=True)
    axes[0].plot(t,zi[0],label="vy"); axes[0].plot(t,zi[1],label="r")
    set_labels(axes[0],"","Response","Zero-input response")
    axes[0].legend()
    axes[1].plot(t,zs[0],label="vy"); axes[1].plot(t,zs[1],label="r")
    set_labels(axes[1],"","Response","Zero-state response")
    axes[1].legend()
    axes[2].plot(t,total[0],label="vy"); axes[2].plot(t,total[1],label="r")
    set_labels(axes[2],"Time (s)","Response","Total response")
    axes[2].legend()
    savefig(fig,os.path.join(FIG_DIR,"15_state_transition_responses.png"))
    fig,ax=plt.subplots()
    ax.plot(t,np.rad2deg([sine_ref(ti) for ti in t]),label="Reference")
    ax.plot(t,np.rad2deg(sine_states[:,1]),label="Actual")
    set_labels(ax,"Time (s)","Yaw rate (deg/s)","Sinusoidal yaw-rate tracking")
    ax.legend(); savefig(fig,os.path.join(FIG_DIR,"19_sinusoidal_tracking.png"))

    pd.DataFrame({"configuration":["full_order","reduced_order"],
                  "tracking_rmse":[np.sqrt(np.mean((step_states[:,1]-combined_full[0][:,1])**2)),
                                   np.sqrt(np.mean((step_states[:,1]-combined_red[0][:,1])**2))],
                  "max_steering_deg":[np.rad2deg(np.max(np.abs(combined_full[2]))),
                                      np.rad2deg(np.max(np.abs(combined_red[2])))]}
                 ).to_csv(os.path.join(TABLE_DIR,"combined_system_performance.csv"),index=False)

    fig,ax=plt.subplots()
    ax.plot(t,np.rad2deg([step_ref(ti) for ti in t]),label="Reference")
    ax.plot(t,np.rad2deg(combined_full[0][:,1]),label="Full-order")
    ax.plot(t,np.rad2deg(combined_red[0][:,1]),label="Reduced-order")
    set_labels(ax,"Time (s)","Yaw rate (deg/s)","Combined tracking comparison")
    ax.legend(); savefig(fig,os.path.join(FIG_DIR,"30_combined_tracking.png"))

    summary={"A":A,"B":B,"eigenvalues":vals,"controllability_rank":rankc,
             "observability_rank":ranko,"K":K,"Ka":Ka,"K_lqr":K_lqr,
             "controller_poles":Kpoles,"ackermann_poles":Kapoles,"lqr_poles":Klqrpoles,
             "L":L,"La":La,"Ll":Ll,"observer_poles":opoles,
             "ackermann_observer_poles":oapoles,"lqr_observer_poles":olpoles,
             "Lr":Lr,"reduced_pole":reduced_pole,
             "step_rmse":step_rmse,"sine_rmse":sine_rmse,
             "max_steering_deg":float(np.max(np.abs(np.rad2deg(step_control)))),
             "controller_table":controller_table.to_dict("records")}
    figures=[("Pacejka Tire Force Characteristics",os.path.join(FIG_DIR,"01_pacejka_tire_forces.png")),
             ("Open-Loop Nonlinear/Linear Response",os.path.join(FIG_DIR,"04_nonlinear_linear_open_loop.png")),
             ("Eigenvalue Plot",os.path.join(FIG_DIR,"09_eigenvalues.png")),
             ("State-Transition Responses",os.path.join(FIG_DIR,"15_state_transition_responses.png")),
             ("Step Tracking",os.path.join(FIG_DIR,"18_step_tracking.png")),
             ("Sinusoidal Tracking",os.path.join(FIG_DIR,"19_sinusoidal_tracking.png")),
             ("Full-Order Observer",os.path.join(FIG_DIR,"23_full_observer_states.png")),
             ("Measurement-Noise RMSE",os.path.join(FIG_DIR,"29_measurement_noise_rmse.png")),
             ("Six-Configuration Step Tracking",os.path.join(FIG_DIR,"30_combined_step_comparison.png")),
             ("Six-Configuration Sinusoidal Tracking",os.path.join(FIG_DIR,"31_combined_sine_comparison.png")),
             ("LQR Full-Order States and Steering",os.path.join(FIG_DIR,"32_lqr_full_states_errors_control.png")),
             ("LQR Reduced-Order States and Steering",os.path.join(FIG_DIR,"32_lqr_reduced_states_errors_control.png"))]
    generate_report(os.path.join(ROOT,"report","project_report.pdf"),summary,figures)
    logging.info("Complete workflow finished; report=%s",
                 os.path.join(ROOT,"report","project_report.pdf"))
    return summary


def main():
    parser=argparse.ArgumentParser(description="AE 695 vehicle dynamics project")
    parser.add_argument("--analysis",choices=["complete","report"],default="complete")
    args=parser.parse_args()
    start=time.time()
    try:
        if args.analysis=="report":
            ensure_dirs()
            logging.info("Report-only mode requires previously saved matrices")
        else:
            summary=run_complete()
            print("\nCOMPLETION SUMMARY")
            print("A =",summary["A"])
            print("B =",summary["B"])
            print("Eigenvalues =",summary["eigenvalues"])
            print("Controllability/observability =",summary["controllability_rank"],summary["observability_rank"])
            print("Controller gain =",summary["K"])
            print("Observer gain =",summary["L"])
            print("Elapsed seconds =",round(time.time()-start,2))
    except Exception as exc:
        logging.exception("Critical failure: %s",exc)
        return 1
    return 0


if __name__=="__main__":
    sys.exit(main())
