// Hound -- A WAV encoding and decoding library in Rust
// Copyright (C) 2015 Ruud van Asseldonk
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License, version 3,
// as published by the Free Software Foundation.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

// This example computes the mean value and rms of a file, where samples are
// first interpreted as 16-bit signed integer, and then as a 16-bit unsigned
// integer. This should allow us to determine whether the samples stored are
// signed or unsigned: for signed the average value is expected to be 0, and
// for unsigned the value is expected to be 2^16 - 1.
//   Note that this example is not a particularly good example of proper coding
// style; it does not handle failure properly, and it assumes that the provided
// file has 16 bits per sample.

// TODO: This example should probably be removed, it is just here for verifying
// and assumption at this point.

extern crate hound;

use std::env;

fn main() {
    let fname = env::args().nth(1).expect("no file given");
    let mut reader = hound::WavReader::open(&fname).unwrap();
    let samples: Vec<i16> = reader.samples().map(|s| s.unwrap()).collect();

    let (ts, tu, n) = samples.iter().fold((0.0, 0.0, 0.0), |(ts, tu, n), &s| {
        let signed = s as f64;
        let unsigned = (s as u16) as f64;
        (ts + signed, tu + unsigned, n + 1.0)
    });
    let ms = ts / n;
    let mu = tu / n;
    println!("mean signed:   {} (should be 0, deviation is {})", ms, ms.abs());
    println!("mean unsigned: {} (should be 2^16 - 1, deviation is {})", mu, (mu - 32767.0).abs());

    let (ts, tu) = samples.iter().fold((0.0, 0.0), |(ts, tu), &s| {
        let ds = s as f64 - ms;
        let du = (s as u16) as f64 - mu;
        (ts + ds * ds, tu + du * du)
    });
    let rmss = (ts / n).sqrt();
    let rmsu = (tu / n).sqrt();
    println!("rms signed:    {}", rmss);
    println!("rms unsigned:  {}", rmsu);
}
