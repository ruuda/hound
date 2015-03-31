// Hound -- A WAV encoding library in Rust
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

//! Hound, a WAV encoding library.
//!
//! TODO: Add some introductory text here.
//!
//! The following example renders a 440 Hz sine wave, and stores it as as a
//! mono wav file with a sample rate of 44.1 kHz and 16 bits per sample.
//!
//! ```
//! #![feature(core)]
//!
//! use std::fs;
//! use std::f32::consts::PI;
//! use std::iter::IntoIterator;
//! use std::num;
//! use hound;
//!
//! let spec = hound::WavSpec {
//!     channels: 1,
//!     sample_rate: 44100,
//!     bits_per_sample: hound::BitDepth::Bps16
//! };
//! // TODO: ensure that the type can be inferred.
//! let mut writer = hound::WavWriter::<fs::File>::create("sine.wav", spec);
//! let mut writer = writer.ok().unwrap();
//! for t in (0 .. 44100).into_iter().map(|x| x as f32 / 44100.0) {
//!     let sample = (t * 440.0 * 2.0 * PI).sin();
//!     let amplitude: i16 = num::Int::max_value();
//!     writer.write_sample((sample * amplitude as f32) as i16).ok().unwrap();
//! }
//! writer.finalize().ok().unwrap();
//! ```

#![warn(missing_docs)]
#![allow(dead_code)] // TODO: Remove for v0.1
#![feature(convert, core, io)]

use std::fs;
use std::io;
use std::io::{Seek, Write};
use std::num;
use std::path;

/// Extends the functionality of `io::Write` with additional methods.
///
/// The methods may be used on any type that implements `io::Write`.
trait WriteExt: io::Write {
    /// Writes an unsigned 16-bit integer in little endian format.
    fn write_le_u16(&mut self, x: u16) -> io::Result<()>;

    /// Writes an unsigned 32-bit integer in little endian format.
    fn write_le_u32(&mut self, x: u32) -> io::Result<()>;
}

impl<W> WriteExt for W where W: io::Write {

    fn write_le_u16(&mut self, x: u16) -> io::Result<()> {
        let mut buf = [0u8; 2];
        buf[0] = (x & 0xff) as u8;
        buf[1] = (x >> 8) as u8;
        self.write_all(&buf)
    }

    fn write_le_u32(&mut self, x: u32) -> io::Result<()> {
        let mut buf = [0u8; 4];
        buf[0] = ((x >> 00) & 0xff) as u8;
        buf[1] = ((x >> 08) & 0xff) as u8;
        buf[2] = ((x >> 16) & 0xff) as u8;
        buf[3] = ((x >> 24) & 0xff) as u8;
        self.write_all(&buf)
    }
}

/// A type that can be used to represent audio samples.
pub trait Sample {
    /// Writes the audio sample to the WAVE data section.
    fn write<W: io::Write>(self, writer: &mut W, bits: u32) -> io::Result<()>;
}

impl Sample for u16 {
    fn write<W: io::Write>(self, writer: &mut W, bits: u32) -> io::Result<()> {
        writer.write_le_u16(self)
        // TODO: take bits into account
    }
}

impl Sample for i16 {
    fn write<W: io::Write>(self, writer: &mut W, bits: u32) -> io::Result<()> {
        writer.write_le_u16(self as u16)
        // TODO: take bits into account
    }
}

/// The number of bits per sample, as a multiple of 8.
#[derive(Clone, Copy)]
pub enum BitDepth {
    /// 8 bits per sample.
    Bps8,

    /// 16 bits per sample.
    Bps16,

    /// 24 bits per sample.
    Bps24,

    /// 32 bits per sample.
    Bps32
}

impl BitDepth {
    /// Returns the number of bits per sample as an integer.
    pub fn into_u32(self) -> u32 {
        match self {
            BitDepth::Bps8 => 8,
            BitDepth::Bps16 => 16,
            BitDepth::Bps24 => 24,
            BitDepth::Bps32 => 32
        }
    }
}


impl num::FromPrimitive for BitDepth {
    fn from_u8(n: u8) -> Option<BitDepth> {
        match n {
            8 => Some(BitDepth::Bps8),
            16 => Some(BitDepth::Bps16),
            24 => Some(BitDepth::Bps24),
            32 => Some(BitDepth::Bps32),
            _ => None
        }
    }

    fn from_u16(n: u16) -> Option<BitDepth> { num::from_u8(n as u8) }
    fn from_u32(n: u32) -> Option<BitDepth> { num::from_u8(n as u8) }
    fn from_u64(n: u64) -> Option<BitDepth> { num::from_u8(n as u8) }
    fn from_i16(n: i16) -> Option<BitDepth> { num::from_u8(n as u8) }
    fn from_i32(n: i32) -> Option<BitDepth> { num::from_u8(n as u8) }
    fn from_i64(n: i64) -> Option<BitDepth> { num::from_u8(n as u8) }
    fn from_f32(n: f32) -> Option<BitDepth> { num::from_u8(n as u8) }
    fn from_f64(n: f64) -> Option<BitDepth> { num::from_u8(n as u8) }
    fn from_usize(n: usize) -> Option<BitDepth> { num::from_u8(n as u8) }
    fn from_isize(n: isize) -> Option<BitDepth> { num::from_u8(n as u8) }
}

impl num::ToPrimitive for BitDepth {
    fn to_u8(&self) -> Option<u8> { Some(self.into_u32() as u8) }
    fn to_u16(&self) -> Option<u16> { Some(self.into_u32() as u16) }
    fn to_u32(&self) -> Option<u32> { Some(self.into_u32() as u32) }
    fn to_u64(&self) -> Option<u64> { Some(self.into_u32() as u64) }
    fn to_i16(&self) -> Option<i16> { Some(self.into_u32() as i16) }
    fn to_i32(&self) -> Option<i32> { Some(self.into_u32() as i32) }
    fn to_i64(&self) -> Option<i64> { Some(self.into_u32() as i64) }
    fn to_f32(&self) -> Option<f32> { Some(self.into_u32() as f32) }
    fn to_f64(&self) -> Option<f64> { Some(self.into_u32() as f64) }
    fn to_usize(&self) -> Option<usize> { Some(self.into_u32() as usize) }
    fn to_isize(&self) -> Option<isize> { Some(self.into_u32() as isize) }
}

impl num::NumCast for BitDepth {
    fn from<T>(n: T) -> Option<BitDepth> where T: num::ToPrimitive {
        n.to_u8().and_then(num::FromPrimitive::from_u8)
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
    pub bits_per_sample: BitDepth
}

/// A writer that accepts samples and writes the WAVE format.
///
/// TODO: add example.
///
/// After all samples have been written, the file must be finalized. This can
/// be done by calling `finalize`. If `finalize` is not called, the file will
/// be finalized upon drop. However, finalization involves IO that may fail,
/// and without calling `finalize`, such a failure cannot be observed.
pub struct WavWriter<W> where W: io::Write + io::Seek {
    /// Specifies properties of the audio data.
    spec: WavSpec,

    /// Whether the header has been written already.
    wrote_header: bool,

    /// The writer that will be written to.
    writer: io::BufWriter<W>,

    /// The number of bytes written to the data section.
    ///
    /// This is an `u32` because WAVE cannot accomodate more data.
    data_bytes_written: u32,

    /// Whether `finalize_internal` has been called.
    finalized: bool
}

impl<W> WavWriter<W> where W: io::Write + io::Seek {
    /// Creates a writer that writes the WAVE format to the underlying writer.
    ///
    /// The underlying writer is assumed to be at offset 0. `WavWriter` employs
    /// buffering internally to avoid too many `write` calls to the underlying
    /// writer.
    pub fn new(writer: W, spec: WavSpec) -> WavWriter<W> {
        WavWriter {
            spec: spec,
            wrote_header: false,
            writer: io::BufWriter::new(writer),
            data_bytes_written: 0,
            finalized: false
        }
    }

    /// Creates a writer that writes the WAVE format to a file.
    ///
    /// The file will be overwritten if it exists.
    pub fn create<P: AsRef<path::Path>>(filename: P, spec: WavSpec)
           -> io::Result<WavWriter<fs::File>> {
        let file = try!(fs::File::create(filename));
        Ok(WavWriter::new(file, spec))
    }

    /// Writes the RIFF WAVE header
    fn write_header(&mut self) -> io::Result<()> {
        let mut header = [0u8; 44];
        let spec = &self.spec;
        let bps = spec.bits_per_sample.into_u32();

        // Write the header in-memory first.
        {
            let mut buffer: io::Cursor<&mut [u8]> = io::Cursor::new(&mut header);
            try!(buffer.write_all("RIFF".as_bytes()));

            // Skip 4 bytes that will be filled with the file size afterwards.
            try!(buffer.seek(io::SeekFrom::Current(4)));

            try!(buffer.write_all("WAVE".as_bytes()));
            try!(buffer.write_all("fmt\0".as_bytes()));
            try!(buffer.write_le_u32(16)); // Size of the WAVE header
            try!(buffer.write_le_u16(1));  // PCM encoded audio
            try!(buffer.write_le_u16(spec.channels));
            try!(buffer.write_le_u32(spec.sample_rate));
            let bytes_per_sec = spec.sample_rate
                              * bps
                              * spec.channels as u32 / 8;
            try!(buffer.write_le_u32(bytes_per_sec));
            try!(buffer.write_le_u16(16)); // TODO: block align
            try!(buffer.write_le_u32(bps));

            // TODO: data section header
        }

        // Then write the entire header at once.
        try!(self.writer.write_all(&header));

        Ok(())
    }

    /// Writes a single sample for one channel.
    ///
    /// WAVE interleaves channel data, so the channel that this writes the
    /// sample to depends on previous writes.
    pub fn write_sample<S: Sample>(&mut self, sample: S) -> io::Result<()> {
        if !self.wrote_header {
            try!(self.write_header());
        }

        let bps = self.spec.bits_per_sample.into_u32();
        try!(sample.write(&mut self.writer, bps));
        self.data_bytes_written += bps / 8;
        Ok(())
    }

    /// Performs finalization. After calling this, the writer should be destructed.
    fn finalize_internal(&mut self) -> io::Result<()> {
        self.finalized = true;

        // Flush remaining samples via the BufWriter.
        try!(self.writer.flush());

        // Extract the underlying writer and rewind it to the start, to update
        // the header fields of which we now know the value.
        let mut writer = self.writer.get_mut();
        try!(writer.seek(io::SeekFrom::Start(0)));

        // TODO: update size fields
        Ok(())
    }

    /// Writes the parts of the WAVE format that require knowing all samples.
    ///
    /// This method must be called after all samples have been written. If it
    /// is not called, the destructor will finalize the file, but any IO errors
    /// that occur in the process cannot be observed in that manner.
    pub fn finalize(mut self) -> io::Result<()> {
        self.finalize_internal()
    }
}

impl<W> Drop for WavWriter<W> where W: io::Write + io::Seek {
    fn drop(&mut self) {
        // `finalize_internal` must be called only once. If that is done via
        // `finalize`, then this method is a no-op. If the user did not
        // finalize explicitly, then we should do it now. This can fail, but
        // drop should not panic, so a failure is ignored silently here.
        if !self.finalized {
            let _r = self.finalize_internal();
        }
    }
}
