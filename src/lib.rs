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

//! Hound, a WAV encoding and decoding library.
//!
//! TODO: Add some introductory text here.
//!
//! The following example renders a 440 Hz sine wave, and stores it as as a
//! mono wav file with a sample rate of 44.1 kHz and 16 bits per sample.
//!
//! ```
//! use std::fs;
//! use std::f32::consts::PI;
//! use std::i16;
//! use hound;
//!
//! let spec = hound::WavSpec {
//!     channels: 1,
//!     sample_rate: 44100,
//!     bits_per_sample: 16
//! };
//! // TODO: ensure that the type can be inferred.
//! let writer_res = hound::WavWriter::<fs::File>::create("sine.wav", spec);
//! let mut writer = writer_res.ok().unwrap();
//! for t in (0 .. 44100).map(|x| x as f32 / 44100.0) {
//!     let sample = (t * 440.0 * 2.0 * PI).sin();
//!     let amplitude: i16 = i16::MAX;
//!     writer.write_sample((sample * amplitude as f32) as i16).ok().unwrap();
//! }
//! writer.finalize().ok().unwrap();
//! ```

#![warn(missing_docs)]

use std::io;
use std::io::Write;
use read::ReadExt;
use write::WriteExt;

mod read;
mod write;

pub use read::{WavReader, WavSamples};
pub use write::WavWriter;

/// A type that can be used to represent audio samples.
pub trait Sample {
    /// Writes the audio sample to the WAVE data chunk.
    fn write<W: io::Write>(self, writer: &mut W, bits: u32) -> io::Result<()>;

    /// Reads the audio sample from the WAVE data chunk.
    fn read<R: io::Read>(reader: &mut R, bits: u32) -> io::Result<Self>;
}

impl Sample for i16 {
    fn write<W: io::Write>(self, writer: &mut W, bits: u32) -> io::Result<()> {
        writer.write_le_i16(self)
        // TODO: take bits into account.
    }

    fn read<R: io::Read>(reader: &mut R, bits: u32) -> io::Result<i16> {
        reader.read_le_i16()
        // TODO: take bits into account.
    }
}

/// Specifies properties of the audio data.
#[derive(Clone, Copy)]
pub struct WavSpec {
    /// The number of channels.
    pub channels: u16,

    /// The number of samples per second.
    ///
    /// A common value is 44100, this is 44.1 kHz which is used for CD audio.
    pub sample_rate: u32,

    /// The number of bits per sample.
    ///
    /// A common value is 16 bits per sample, which is used for CD audio.
    pub bits_per_sample: u32
}
