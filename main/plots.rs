use crate::{arg_of_ones, find_edges, load_clocks, load_power_traces, resample_traces};
use ndarray::{s, Array1, ArrayView1, ArrayView2};
use plotly::Plot;
use std::iter::zip;

pub fn plot_power_traces(trace: &ArrayView2<f64>) {
    let mut plot = Plot::new();

    for row in trace.rows().into_iter().enumerate() {
        let (i, trace) = row;

        let x_indices: Vec<usize> = (0..trace.len()).collect();

        use plotly::common::Mode;

        let power_trace = plotly::Scatter::new(x_indices.clone(), trace.to_vec())
            .name(format!("Power {i}"))
            .mode(Mode::Lines);

        plot.add_trace(power_trace);
    }

    plot.show();
}

pub fn plot_multiple_power_traces_with_clocks(trace: &ArrayView2<f64>, clock: &ArrayView2<f64>) {
    let mut plot = Plot::new();

    for row in zip(trace.rows(), clock.rows()).enumerate() {
        let (i, (trace, clock)) = row;

        let x_indices: Vec<usize> = (0..trace.len()).collect();

        use plotly::common::Mode;

        let power_trace = plotly::Scatter::new(x_indices.clone(), trace.to_vec())
            .name(format!("Power {i}"))
            .mode(Mode::Lines);

        let clock_trace = plotly::Scatter::new(x_indices, clock.to_vec())
            .name(format!("Clock {i}"))
            .mode(Mode::Lines);

        plot.add_trace(power_trace);
        plot.add_trace(clock_trace);
    }

    plot.show();
}

pub fn plot_resampled_traces_with_reference_clock(
    ref_clock: &Array1<f64>,
    original: &ArrayView1<f64>,
    resampled_power: &ArrayView1<f64>,
    resampled_clock: &ArrayView1<f64>,
) {
    let mut plot = Plot::new();

    let x_indices: Vec<usize> = (0..original.len()).collect();

    use plotly::common::Mode;

    let ref_clock_trace = plotly::Scatter::new(x_indices.clone(), ref_clock.to_vec())
        .name("Reference clock")
        .fill_color("#00FF00")
        .mode(Mode::Lines);

    let power_trace = plotly::Scatter::new(x_indices.clone(), original.to_vec())
        .name("Original")
        .mode(Mode::Lines);

    let resampled_power_traces = plotly::Scatter::new(x_indices.clone(), resampled_power.to_vec())
        .name("Resampled power")
        .mode(Mode::Lines);

    let resampled_clock_traces = plotly::Scatter::new(x_indices, resampled_clock.to_vec())
        .name("Resampled Clock")
        .mode(Mode::Lines);

    plot.add_trace(ref_clock_trace);
    plot.add_trace(power_trace);
    plot.add_trace(resampled_power_traces);
    plot.add_trace(resampled_clock_traces);
    plot.show()
}

pub fn plot_dataset2(dataset: &str) {
    let power_traces = load_power_traces(dataset);
    let clocks = load_clocks(dataset);

    let ref_clock = clocks.slice(s![0, 0, ..]).to_owned();
    let ref_edges = find_edges(&ref_clock);
    let ref_indices = arg_of_ones(&ref_edges);

    plot_power_traces(&power_traces.slice(s![0, 45..50, 0..500]));

    plot_multiple_power_traces_with_clocks(
        &power_traces.slice(s![0, 45..50, 0..500]),
        &clocks.slice(s![0, 45..50, 0..500]),
    );

    {
        let resampled_power_traces = resample_traces(&ref_indices, &power_traces, &clocks, 2);
        let resampled_clocks = resample_traces(&ref_indices, &clocks, &clocks, 2);
        plot_resampled_traces_with_reference_clock(
            &ref_clock,
            &power_traces.slice(s![0, 2, 0..500]),
            &resampled_power_traces.slice(s![0, 2, 0..500]),
            &resampled_clocks.slice(s![0, 2, 0..500]),
        );
        plot_power_traces(&resampled_power_traces.slice(s![0, 0..50, 0..500]));
    }

    {
        let resampled_power_traces = resample_traces(&ref_indices, &power_traces, &clocks, 20);
        let resampled_clocks = resample_traces(&ref_indices, &clocks, &clocks, 20);

        plot_resampled_traces_with_reference_clock(
            &ref_clock,
            &power_traces.slice(s![0, 2, 0..500]),
            &resampled_power_traces.slice(s![0, 2, 0..500]),
            &resampled_clocks.slice(s![0, 2, 0..500]),
        );
        plot_power_traces(&resampled_power_traces.slice(s![0, 0..50, 0..500]));
    }
}

pub fn plot_dataset1(dataset: &str) {
    let power_traces = load_power_traces(dataset);

    plot_power_traces(&power_traces.slice(s![0, 40..50, 0..500]));
}
