"""Generate a reproducible technical report PDF using ReportLab."""
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, Image, PageBreak)
import os
import pandas as pd


def generate_report(path, summary, figures):
    """Write a concise genuine engineering report containing computed results."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    doc=SimpleDocTemplate(path,pagesize=A4,
                          rightMargin=18*mm,leftMargin=18*mm,
                          topMargin=18*mm,bottomMargin=18*mm)
    styles=getSampleStyleSheet()
    project_root=os.path.dirname(os.path.dirname(path))
    results_root=os.path.join(project_root,"results","tables")
    combined_path=os.path.join(results_root,"controller_observer_comparison.csv")
    noise_path=os.path.join(results_root,"measurement_noise.csv")
    combined_df=pd.read_csv(combined_path) if os.path.exists(combined_path) else pd.DataFrame()
    noise_df=pd.read_csv(noise_path) if os.path.exists(noise_path) else pd.DataFrame()
    canonical_path=os.path.join(results_root,"canonical_validation.csv")
    transition_path=os.path.join(project_root,"results","data","state_transition_responses.csv")
    canonical_df=pd.read_csv(canonical_path) if os.path.exists(canonical_path) else pd.DataFrame()
    story=[Paragraph("AE 695: Vehicle Lateral Dynamics",styles["Title"]),
           Spacer(1,8*mm),
           Paragraph("Mathematical Modeling, Stability Analysis, State Feedback Control and Observer Design",
                     styles["Heading2"]),
           Spacer(1,5*mm),
           Paragraph("Generated automatically from the reproducible Python project.",styles["BodyText"]),
           PageBreak(),Paragraph("Abstract",styles["Heading1"]),
           Paragraph("This report presents a nonlinear dynamic bicycle model with a simplified Pacejka "
                     "Magic Formula tire model. It includes equilibrium and Jacobian analysis, open-loop "
                     "linear and nonlinear validation, state-space control and observer design, measurement "
                     "noise, nonlinear closed-loop simulation and sensitivity studies.",styles["BodyText"]),
           Paragraph("1. Physical Model",styles["Heading1"]),
           Paragraph("The state is [v_y, r]^T and the input is front steering angle. The nonlinear "
                     "equations use tire slip angles, Pacejka lateral forces, force balance and yaw moment balance.",
                     styles["BodyText"]),
           Paragraph("2. Linearized Model and Stability",styles["Heading1"]),
           Paragraph(f"A = {summary['A'].tolist()}<br/>B = {summary['B'].tolist()}<br/>"
                     f"Eigenvalues = {summary['eigenvalues'].tolist()}<br/>"
                     f"Controllability rank = {summary['controllability_rank']}; "
                     f"Observability rank = {summary['observability_rank']}.",
                     styles["BodyText"]),
           Paragraph("3. Controller and Observer Designs",styles["Heading1"]),
           Paragraph(f"Pole-placement gain = {summary['K'].tolist()}<br/>"
                     f"Pole-placement poles = {summary['controller_poles'].tolist()}<br/>"
                     f"Ackermann gain = {summary['Ka'].tolist()}<br/>"
                     f"Ackermann poles = {summary['ackermann_poles'].tolist()}<br/>"
                     f"LQR gain = {summary['K_lqr'].tolist()}<br/>"
                     f"LQR poles = {summary['lqr_poles'].tolist()}<br/>"
                     f"Full-order pole-placement observer gain = {summary['L'].tolist()}<br/>"
                     f"Full-order observer poles = {summary['observer_poles'].tolist()}<br/>"
                     f"Reduced-order gain = {summary['Lr']:.6g}; error pole = {summary['reduced_pole']:.6g}.",
                     styles["BodyText"]),
           Paragraph("4. Tracking and Noise Validation",styles["Heading1"]),
           Paragraph("The following headline metrics are explicitly for the "
                     "pole-placement controller (not general project-wide results): "
                     f"step RMSE = {summary['step_rmse']:.6g} rad/s; "
                     f"sinusoidal RMSE = {summary['sine_rmse']:.6g} rad/s; "
                     f"maximum steering angle = {summary['max_steering_deg']:.3f} degrees.",
                     styles["BodyText"]),
           Paragraph("Controller comparison",styles["Heading2"]),
           Table([[ "Controller","Gain vy","Gain r","Step RMSE (rad)",
                    "Sine RMSE (rad)","Max steering (deg)"]]+[
               [row["method"],f"{row['gain_vy']:.6g}",f"{row['gain_r']:.6g}",
                f"{row['step_rmse_rad']:.6g}",f"{row['sine_rmse_rad']:.6g}",
                f"{row['max_steering_deg']:.3f}"]
               for row in summary.get("controller_table",[])
           ],style=TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.grey),
                               ("BACKGROUND",(0,0),(-1,0),colors.lightgrey),
                               ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
                               ("FONTSIZE",(0,0),(-1,-1),7)])),
           Paragraph("Combined controller-observer comparison",styles["Heading2"]),
           Table([["Configuration","Reference","Tracking RMSE (rad)",
                   "vy estimate RMSE (m/s)","r estimate RMSE (rad/s)",
                   "Max steer (deg)","Sat. duration (s)"]]+[
               [f"{r.configuration}-{r.observer}",str(r.reference),
                f"{r.tracking_rmse_rad:.6g}",f"{r.vy_estimation_rmse_m_s:.6g}",
                f"{r.r_estimation_rmse_rad_s:.6g}",f"{r.max_steering_deg:.3f}",
                f"{r.saturation_duration_s:.3f}"]
               for r in combined_df.itertuples()
           ] if not combined_df.empty else [["No combined results available"]],
           style=TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.grey),
                             ("BACKGROUND",(0,0),(-1,0),colors.lightgrey),
                             ("FONTSIZE",(0,0),(-1,-1),6)])),
           Paragraph("Measurement-noise sensitivity",styles["Heading2"]),
           Table([["Observer","Noise std. (rad/s)","vy RMSE (m/s)",
                   "r RMSE (rad/s)"]]+[
               [str(r.observer),f"{r.sigma_rad_s:.3g}",f"{r.vy_rmse:.6g}",
                f"{r.r_rmse:.6g}"] for r in noise_df.itertuples()
           ] if not noise_df.empty else [["No noise results available"]],
           style=TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.grey),
                             ("BACKGROUND",(0,0),(-1,0),colors.lightgrey),
                             ("FONTSIZE",(0,0),(-1,-1),7)])),
           Paragraph("The reduced-order observer reconstructs lateral velocity as "
                     "vy_hat = z + Lr*r_measured. Its increased noise sensitivity follows "
                     "from direct algebraic dependence on the noisy yaw measurement.",
                     styles["BodyText"]),
           Paragraph("Canonical and state-transition validation",styles["Heading2"]),
           Table([["Realization","Condition","Residual","Rank"]]+[
               [str(r.form),f"{r.T_condition:.6g}",f"{r.residual:.3e}",
                str(r.rank)] for r in canonical_df.itertuples()
           ] if not canonical_df.empty else [["No canonical results available"]],
           style=TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.grey),
                             ("BACKGROUND",(0,0),(-1,0),colors.lightgrey),
                             ("FONTSIZE",(0,0),(-1,-1),7)])),
           Paragraph("Complex modal diagonalization uses eigenvectors as a complex "
                     "similarity transformation. The real modal representation uses real "
                     "and imaginary parts of a complex eigenvector pair. It is not labeled "
                     "a Jordan form. For the constant-input case, zero-input response is "
                     "exp(A t)x0, zero-state response is integral exp(A(t-tau))B u dtau, "
                     "and total response is their sum.",
                     styles["BodyText"]),
           Paragraph("5. Assumptions and Limitations",styles["Heading1"]),
           Paragraph("Constant longitudinal speed, no load transfer, no combined-slip relaxation, no roll/pitch "
                     "and a simplified zero-shift Pacejka model are assumed. Results are local to the selected "
                     "operating region and tire parameters are stiffness-matched rather than measured.",
                     styles["BodyText"]),
           PageBreak()]
    for title,path in figures:
        if os.path.exists(path):
            story.append(Paragraph(title,styles["Heading2"]))
            im=Image(path,width=170*mm,height=105*mm,kind='proportional')
            story.append(im); story.append(Spacer(1,4*mm))
    doc.build(story)
