from pathlib import Path


def test_lateral_velocity_plot_keeps_si_units():
    main_source = Path("main.py").read_text(encoding="utf-8")
    assert 'axes[0].plot(t,states[:,0]' in main_source
    assert 'axes[0].plot(t,estimates[:,0]' in main_source
    assert 'np.rad2deg(states[:,0])' not in main_source
    assert 'np.rad2deg(estimates[:,0])' not in main_source


def test_report_labels_pole_metrics_explicitly():
    report_source = Path("report/generate_report.py").read_text(encoding="utf-8")
    assert "pole-placement controller" in report_source
    assert "Controller comparison" in report_source
