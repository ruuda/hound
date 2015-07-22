// Hound -- A wav encoding and decoding library in Rust
// Copyright (C) 2015 Ruud van Asseldonk
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// A copy of the License has been included in the root of the repository.
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This example shows how to play a wav file using the cpal crate.

extern crate hound;
extern crate cpal;

use std::env;
use std::thread;

fn main() {
    // Make a WavReader that reads the file provided as program argument.
    let fname = env::args().nth(1).expect("no file given");
    let mut reader = hound::WavReader::open(fname).unwrap();
    let spec = reader.spec();

    // A voice in cpal is used for playback.
    let mut voice = cpal::Voice::new();

    let mut append_data = |voice: &mut cpal::Voice| {
        let mut samples = reader.samples::<i16>();
        let samples_left = samples.len();

        if samples_left == 0 { return false; }

        let mut buffer: cpal::Buffer<i16> =
            voice.append_data(spec.channels,
                              cpal::SamplesRate(spec.sample_rate),
                              samples_left);
        // Fill the cpal buffer with data from the wav file.
        for (dest, src) in buffer.iter_mut().zip(&mut samples) {
            *dest = src.unwrap();
        }

        // Probably not done, loop again.
        true
    };

    // The voice must have some data before playing for the first time.
    append_data(&mut voice);
    voice.play();

    // Then we keep providing new data until the end of the audio.
    while append_data(&mut voice) { }

    // TODO: Cpal has no function (yet) to wait for playback to complete, so
    // sleep manually.
    thread::sleep_ms(1000);
}
