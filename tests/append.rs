// Hound -- A wav encoding and decoding library in Rust
// Copyright 2018 Ruud van Asseldonk
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// A copy of the License has been included in the root of the repository.
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::fs;

extern crate hound;

fn assert_contents(fname: &str, expected: &[i16]) {
    let mut reader = hound::WavReader::open(fname).unwrap();
    let samples: Vec<i16> = reader.samples().map(|s| s.unwrap()).collect();
    assert_eq!(&samples[..], expected);
}

#[test]
fn append_works_on_files_not_just_in_memory() {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 44100,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create("append.wav", spec).unwrap();
    writer.write_sample(11_i16).unwrap();
    writer.write_sample(13_i16).unwrap();
    writer.write_sample(17_i16).unwrap();
    writer.finalize().unwrap();

    assert_contents("append.wav", &[11, 13, 17]);

    let len = fs::metadata("append.wav").unwrap().len();

    let mut appender = hound::WavWriter::append("append.wav").unwrap();

    appender.write_sample(19_i16).unwrap();
    appender.write_sample(23_i16).unwrap();
    appender.finalize().unwrap();

    // We appended four bytes of audio data (2 16-bit samples), so the file
    // should have grown by 4 bytes.
    assert_eq!(fs::metadata("append.wav").unwrap().len(), len + 4);

    assert_contents("append.wav", &[11, 13, 17, 19, 23]);
}
