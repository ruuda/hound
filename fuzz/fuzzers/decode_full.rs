// Hound -- A wav encoding and decoding library in Rust
// Copyright (C) 2017 Ruud van Asseldonk
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// A copy of the License has been included in the root of the repository.

#![no_main]

extern crate libfuzzer_sys;
extern crate hound;

use std::io::Cursor;

#[export_name="rust_fuzzer_test_input"]
pub extern fn go(data: &[u8]) {
    let cursor = Cursor::new(data);
    let mut reader = match hound::WavReader::new(cursor) {
        Ok(r) => r,
        Err(..) => return,
    };

    // TODO: For some reason, the iterator-based approach crashes with an
    // obscure memory error, but the while-let-based method works perfectly
    // fine.
    // for sample in reader.samples::<i32>() {
    //     match sample {
    //         Ok(..) => { }
    //         Err(..) => return,
    //     }
    // }

    match reader.spec().sample_format {
        hound::SampleFormat::Int => {
            let mut iter = reader.samples::<i32>();
            while let Some(sample) = iter.next() {
                match sample {
                    Ok(..) => { }
                    Err(..) => return,
                }
            }
        }
        hound::SampleFormat::Float => {
            let mut iter = reader.samples::<f32>();
            while let Some(sample) = iter.next() {
                match sample {
                    Ok(..) => { }
                    Err(..) => return,
                }
            }
        }
    }
}
