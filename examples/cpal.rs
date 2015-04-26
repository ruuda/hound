// Hound -- A WAV encoding and decoding library in Rust
// Copyright (C) 2015 Ruud van Asseldonk
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3 of the License only.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

// This example shows how to play a wav file using the cpal crate.

extern crate hound;
extern crate cpal;

use std::env;

fn main() {
    // Make a WavReader that reads the file provided as program argument.
    let fname = env::args().nth(1).expect("no file given");
    let mut reader = hound::WavReader::open(fname).unwrap();
    let spec = reader.spec().clone(); // TODO: by value might be better in this case.

    // A voice in cpal is used for playback.
    let mut voice = cpal::Voice::new();

    let mut append_data = |voice: &mut cpal::Voice| {
        let mut buffer: cpal::Buffer<u16> =
            voice.append_data(spec.channels,
                              cpal::SamplesRate(spec.sample_rate),
                              1024); // TODO: Do I know the number of samples?
        // Fill the cpal buffer with data from the wav file.
        for (dest, src) in buffer.iter_mut().zip(reader.samples::<i16>()) {
            // TODO: Without the cast, there are a lot of artefacts. This
            // suggests that samples are actually unsigned. We can find out by
            // computing the mean: if it is around 0i16, then samples are signed.
            // If it is around 32000u16, then samples are unsigned.
            *dest = src.unwrap() as u16;
        }
        // TODO: check when reader is done, then exit.
    };

    // The voice must have some data before playing for the first time.
    append_data(&mut voice);
    voice.play();

    // Then we keep providing new data until the end of the audio.
    loop {
        append_data(&mut voice); // TODO exit when done
    }
}
