// Hound -- A wav encoding and decoding library in Rust
// Copyright 2018 Ruud van Asseldonk
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// A copy of the License has been included in the root of the repository.

#![no_main]

extern crate libfuzzer_sys;
extern crate hound;

use std::io;
use std::fmt::Debug;

fn try_append<S, T>(mut buffer: Vec<u8>, sample_narrow: S, sample_wide: T)
where S: hound::Sample + Copy + Debug + PartialEq,
      T: hound::Sample + Copy + Debug + PartialEq {
    let mut samples_orig: Vec<T> = Vec::new();
    let mut samples_after: Vec<T>;

    // Read samples first.
    {
        let cursor = io::Cursor::new(&mut buffer);
        let mut reader = hound::WavReader::new(cursor).unwrap();
        for r in reader.samples() {
            match r {
                Ok(x) => samples_orig.push(x),
                // If reading fails, then don't attempt to append. We are only
                // concerned with appending to valid files.
                Err(..) => return,
            }
        }
    }

    // Open in append mode and append one sample for each channel.
    {
        let cursor = io::Cursor::new(&mut buffer);
        let mut writer = match hound::WavWriter::new_append(cursor) {
            Ok(w) => w,
            Err(hound::Error::Unsupported) => return,
            Err(err) => panic!("{:?}", err),
        };
        for _ in 0..writer.spec().channels {
            writer.write_sample(sample_narrow).unwrap();
        }
    }

    // Read once more.
    {
        let cursor = io::Cursor::new(buffer);
        let mut reader = hound::WavReader::new(cursor)
            .expect("Reading wav failed after append.");
        samples_after = reader.samples().map(|r| r.unwrap()).collect();
    }

    // Replace NaNs with the given sample.
    for x in &mut samples_orig[..] {
        if *x != *x { *x = sample_wide; }
    }
    for x in &mut samples_after[..] {
        if *x != *x { *x = sample_wide; }
    }

    assert_eq!(&samples_orig[..], &samples_after[..samples_orig.len()]);
    assert_eq!(samples_after[samples_after.len() - 1], sample_wide);
}

#[export_name="rust_fuzzer_test_input"]
pub extern fn go(data: &[u8]) {
    let mut buffer = data.to_vec();
    let spec;
    {
        let cursor = io::Cursor::new(&mut buffer);
        match hound::WavReader::new(cursor) {
            Err(..) => return,
            Ok(reader) => spec = reader.spec(),
        };
    }

    match spec.sample_format {
        hound::SampleFormat::Int => try_append::<i8, i32>(buffer, 41_i8, 41_i32),
        hound::SampleFormat::Float => try_append::<f32, f32>(buffer, 0.41, 0.41),
    }
}
