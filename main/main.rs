#![allow(dead_code)]

mod plots;

use crate::plots::{plot_dataset1, plot_dataset2};
use ndarray::{s, Array, Array1, Array2, Array3};
use rayon::prelude::*;
use std::env;
use std::fs::File;
use std::io::read_to_string;
use std::path::Path;
use std::time::Instant;

const SBOX: [i32; 256] = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
];

fn load_power_traces(dataset: &str) -> Array3<f64> {
    let power_traces = (0..16)
        .into_par_iter()
        .flat_map(|x| {
            let path = format!("{dataset}/trace{x}.txt");
            let path = Path::new(&path);
            let file = File::open(path).unwrap();
            let power_trace = read_to_string(file)
                .unwrap()
                .split_whitespace()
                .map(|x| x.parse::<f64>().unwrap())
                .collect::<Vec<_>>();
            return power_trace;
        })
        .collect::<Vec<_>>();

    Array::from_shape_vec((16, 150, 50000), power_traces).unwrap()
}

fn load_clocks(dataset: &str) -> Array3<f64> {
    let clocks = (0..16)
        .into_par_iter()
        .flat_map(|x| {
            let path = format!("{dataset}/clock{x}.txt");
            let path = Path::new(&path);
            let file = File::open(path).unwrap();
            let clock = read_to_string(file)
                .unwrap()
                .split_whitespace()
                .map(|x| x.parse::<f64>().unwrap())
                .collect::<Vec<_>>();
            return clock;
        })
        .collect::<Vec<_>>();

    Array::from_shape_vec((16, 150, 50000), clocks).unwrap()
}

fn load_clear_text(path: &str) -> Array2<i32> {
    let path = Path::new(path);
    let file = File::open(path).unwrap();
    let cleartext = read_to_string(file)
        .unwrap()
        .split_whitespace()
        .map(|x| x.parse::<i32>())
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    Array::from_shape_vec((cleartext.len() / 16, 16), cleartext).unwrap()
}

fn calculate_hamming_weights(clear_text: &Array2<i32>) -> Array3<f64> {
    let hamming_weights = (0..16)
        .into_par_iter()
        .flat_map(|byte_index| {
            (0..256)
                .flat_map(|key_guess| {
                    let hw = clear_text
                        .column(byte_index)
                        .map(|byte| SBOX[(byte ^ key_guess) as usize])
                        .map(|sbox_output| sbox_output.count_ones() as f64);
                    hw
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    // byte_index, key_guess, hw
    Array3::from_shape_vec((16, 256, 150), hamming_weights).unwrap()
}

fn pearson_correlation(x: &ndarray::ArrayView1<f64>, y: &ndarray::ArrayView1<f64>) -> f64 {
    let mean_x = x.mean().unwrap();
    let mean_y = y.mean().unwrap();

    if mean_x == 0.0 || mean_y == 0.0 {
        return 0.0;
    }

    let diff_x = x - mean_x;
    let diff_y = y - mean_y;

    let numerator = diff_x.dot(&diff_y);

    let denom_x = diff_x.dot(&diff_x).sqrt();
    let denom_y = diff_y.dot(&diff_y).sqrt();

    (numerator / (denom_x * denom_y)).abs()
}

fn find_edges(clock: &Array1<f64>) -> Array1<f64> {
    (1..clock.len())
        .map(|i| {
            if clock[i] > 0.5 && clock[i - 1] < 0.5 {
                1.0
            } else {
                0.0
            }
        })
        .collect()
}

fn arg_of_ones(array: &Array1<f64>) -> Array1<usize> {
    array
        .iter()
        .enumerate()
        .filter_map(|(idx, &val)| if val == 1.0 { Some(idx) } else { None })
        .collect()
}

fn cpa_attack(power_traces: &Array3<f64>, hamming_weights: &Array3<f64>) -> Vec<i32> {
    let key = (0..power_traces.shape()[0])
        .map(|byte_index| {
            // 150 x 50000
            let byte_power = power_traces.slice(s![byte_index, .., ..]);

            // 256 x 150
            let byte_hw = hamming_weights.slice(s![byte_index, .., ..]);

            println!("Calculating byte {byte_index}...");

            // 256 x 1
            let (byte_guess, _byte_correlation) = (0..hamming_weights.shape()[1])
                .into_par_iter()
                .map(|key_idx| {
                    // 150 x 1
                    let hw = byte_hw.slice(s![key_idx, ..]);

                    let max_corr = (0..power_traces.shape()[2])
                        .map(|sample_idx| {
                            // 150 x 1
                            let sample = byte_power.slice(s![.., sample_idx]);
                            pearson_correlation(&sample, &hw)
                        })
                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                        .unwrap();

                    (key_idx, max_corr)
                })
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap();

            println!("byte {} {}", byte_index, byte_guess);
            byte_guess as i32
        })
        .collect::<Vec<_>>();
    key
}

fn resample_traces(
    ref_indices: &Array1<usize>,
    target_trace: &Array3<f64>,
    clocks: &Array3<f64>,
    window_size: usize,
) -> Array3<f64> {
    let resampled_trace = (0..target_trace.shape()[0])
        .into_par_iter()
        .flat_map(|byte| {
            (0..target_trace.shape()[1])
                .into_par_iter()
                .flat_map(|x| {
                    let trace = target_trace.slice(s![byte, x, ..]).to_owned();
                    let clock = clocks.slice(s![byte, x, ..]).to_owned();

                    let resampled_trace = resample(&ref_indices, &trace, &clock, window_size);
                    resampled_trace.to_vec()
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    Array3::from_shape_vec(
        (
            target_trace.shape()[0],
            target_trace.shape()[1],
            resampled_trace.len() / (target_trace.shape()[0] * target_trace.shape()[1]),
        ),
        resampled_trace,
    )
    .unwrap()
}

fn resample(
    ref_indices: &Array1<usize>,
    power_traces: &Array1<f64>,
    clock: &Array1<f64>,
    window_size: usize,
) -> Array1<f64> {
    let clock_edges = find_edges(clock);
    let edge_indices = arg_of_ones(&clock_edges);

    let min_len = std::cmp::min(ref_indices.len(), edge_indices.len());
    let ref_indices = ref_indices.slice(s![..min_len]);
    let edge_indices = edge_indices.slice(s![..min_len]);

    let offsets = edge_indices.map(|x| *x as isize) - ref_indices.map(|x| *x as isize);
    let mut resampled_array = Array1::<f64>::zeros(500 * window_size);

    assert!(
        offsets.len() < resampled_array.len(),
        "Must have less than 500 edges"
    );

    let mut resampled_edges_count = 0;
    let mut edge_count = 0;
    (0..power_traces.len()).for_each(|trace_index| {
        if edge_count >= edge_indices.len() {
            return;
        }

        let index = ref_indices[edge_count];
        let left = index - window_size / 2;
        let right = index + window_size / 2;
        let target_index = trace_index + offsets[edge_count] as usize;

        if left <= 0 || right >= power_traces.len() || target_index >= power_traces.len() {
            return;
        }

        if trace_index >= left && trace_index < right {
            resampled_array[resampled_edges_count] = power_traces[target_index];
            resampled_edges_count += 1;
        }

        if trace_index == right {
            edge_count += 1;
        }
    });

    resampled_array
}

fn attack_ds1(dataset: &str) {
    // 150 x 16
    let clear_text = load_clear_text(&format!("{dataset}/cleartext.txt"));
    // 16 x 256 x 150
    let hamming_weights = calculate_hamming_weights(&clear_text);
    // 16 x 150 x 50000
    let power_traces = load_power_traces(dataset);

    let key = cpa_attack(&power_traces, &hamming_weights);

    dbg!(&key
        .iter()
        .map(|x| format!("0x{x:X}"))
        .collect::<Vec<_>>()
        .join(", "));
    dbg!(key.iter().sum::<i32>());
    assert_eq!(key.iter().sum::<i32>(), 1712);
}

fn attack_ds2(dataset: &str) {
    // 150 x 16
    let clear_text = load_clear_text(&format!("{dataset}/cleartext.txt"));
    // 16 x 256 x 150
    let hamming_weights = calculate_hamming_weights(&clear_text);
    // 16 x 150 x 50000
    let power_traces = load_power_traces(dataset);
    // 16 x 150 x 50000
    let clocks = load_clocks(dataset);

    // 1 x 1 x 50000
    let ref_clock = clocks.slice(s![0, 0, ..]).to_owned();
    let ref_edges = find_edges(&ref_clock);
    let ref_indices = arg_of_ones(&ref_edges);

    let resampled_power_traces = resample_traces(&ref_indices, &power_traces, &clocks, 2);

    let key = cpa_attack(&resampled_power_traces, &hamming_weights);

    dbg!(&key
        .iter()
        .map(|x| format!("0x{x:X}"))
        .collect::<Vec<_>>()
        .join(", "));
    dbg!(key.iter().sum::<i32>());
    assert_eq!(key.iter().sum::<i32>(), 1434);
}
fn main() {
    println!("Thread pool size: {}", rayon::current_num_threads());
    let args = env::args().collect::<Vec<_>>();

    let total_start_time = Instant::now();
    if args.len() == 3 && args[2].eq("plot") {
        if args[1].ends_with("2") {
            plot_dataset2(args[1].as_str());
        }
        if args[1].ends_with("1") {
            plot_dataset1(args[1].as_str());
        }
    }

    if args.len() == 2 {
        if args[1].ends_with("1") {
            attack_ds1(args[1].as_str());
        }

        if args[1].ends_with("2") {
            attack_ds2(args[1].as_str());
        }
    }

    let total_duration = total_start_time.elapsed();
    println!("Total execution time: {:?}", total_duration);
}
