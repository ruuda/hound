// Hound -- A wav encoding and decoding library in Rust
// Copyright (C) 2018 Ruud van Asseldonk
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// A copy of the License has been included in the root of the repository.

#[macro_use] extern crate honggfuzz;

extern crate hound;

use std::io::Cursor;

fn main() {
    loop {
        fuzz!(|data: &[u8]| {

            let cursor = Cursor::new(data);
            let mut reader = match hound::WavReader::new(cursor) {
                Ok(r) => r,
                Err(..) => return,
            };

            match reader.spec().sample_format {
                hound::SampleFormat::Int => {
                    let mut iter = reader.samples::<i32>();
                    while let Some(sample) = iter.next() {
                        match sample {
                            Ok(..) => { }
                            Err(..) => break,
                        }
                    }
                }
                hound::SampleFormat::Float => {
                    let mut iter = reader.samples::<f32>();
                    while let Some(sample) = iter.next() {
                        match sample {
                            Ok(..) => { }
                            Err(..) => break,
                        }
                    }
                }
            }
        });
    }
}
