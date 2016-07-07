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

//! Hound, a wav encoding and decoding library.
//!
//! Examples
//! ========
//!
//! The following example renders a 440 Hz sine wave, and stores it as as a
//! mono wav file with a sample rate of 44.1 kHz and 16 bits per sample.
//!
//! ```
//! use std::f32::consts::PI;
//! use std::i16;
//! use hound;
//!
//! let spec = hound::WavSpec {
//!     channels: 1,
//!     sample_rate: 44100,
//!     bits_per_sample: 16,
//!     sample_format: hound::SampleFormat::Int,
//! };
//! let mut writer = hound::WavWriter::create("sine.wav", spec).unwrap();
//! for t in (0 .. 44100).map(|x| x as f32 / 44100.0) {
//!     let sample = (t * 440.0 * 2.0 * PI).sin();
//!     let amplitude = i16::MAX as f32;
//!     writer.write_sample((sample * amplitude) as i16).unwrap();
//! }
//! writer.finalize().unwrap();
//! ```
//!
//! The following example computes the root mean square (RMS) of an audio file
//! with at most 16 bits per sample.
//!
//! ```
//! use hound;
//!
//! let mut reader = hound::WavReader::open("testsamples/pop.wav").unwrap();
//! let sqr_sum = reader.samples::<i16>()
//!                     .fold(0.0, |sqr_sum, s| {
//!     let sample = s.unwrap() as f64;
//!     sqr_sum + sample * sample
//! });
//! println!("RMS is {}", (sqr_sum / reader.len() as f64).sqrt());
//! ```

#![warn(missing_docs)]

use std::error;
use std::fmt;
use std::io;
use std::result;
use read::ReadExt;
use write::WriteExt;

mod read;
mod write;

pub use read::{WavReader, WavIntoSamples, WavSamples};
pub use write::WavWriter;

/// A type that can be used to represent audio samples.
///
/// Via this trait, decoding can be generic over `i8`, `i16`, `i32` and `f32`.
///
/// All integer formats with bit depths up to 32 bits per sample can be decoded
/// into `i32`, but it takes up more memory. If you know beforehand that you
/// will be reading a file with 16 bits per sample, then decoding into an `i16`
/// will be sufficient.
pub trait Sample: Sized {
    /// Writes the audio sample to the WAVE data chunk.
    fn write<W: io::Write>(self, writer: &mut W, bits: u16) -> Result<()>;

    /// Reads the audio sample from the WAVE data chunk.
    fn read<R: io::Read>(reader: &mut R, SampleFormat, bytes: u16, bits: u16) -> Result<Self>;
}

/// Converts an unsigned integer in the range 0-255 to a signed one in the range -128-127.
///
/// Presumably, the designers of the WAVE format did not like consistency. For
/// all bit depths except 8, samples are stored as little-endian _signed_
/// integers. However, an 8-bit sample is instead stored as an _unsigned_
/// integer. Hound abstracts away this idiosyncrasy by providing only signed
/// sample types.
fn signed_from_u8(x: u8) -> i8 {
    (x as i16 - 128) as i8
}

/// Converts a signed integer in the range -128-127 to an unsigned one in the range 0-255.
fn u8_from_signed(x: i8) -> u8 {
    (x as i16 + 128) as u8
}

#[test]
fn u8_sign_conversion_is_bijective() {
    for x in 0 .. 255 {
        assert_eq!(x, u8_from_signed(signed_from_u8(x)));
    }
    for x in -128 .. 127 {
        assert_eq!(x, signed_from_u8(u8_from_signed(x)));
    }
}

/// Tries to cast the sample to an 8-bit signed integer, returning an error on overflow.
fn narrow_to_i8(x: i32) -> Result<i8> {
    use std::i8;
    if x < i8::MIN as i32 || x > i8::MAX as i32 {
        Err(Error::TooWide)
    } else {
        Ok(x as i8)
    }
}

#[test]
fn verify_narrow_to_i8() {
    assert!(narrow_to_i8(127).is_ok());
    assert!(narrow_to_i8(128).is_err());
    assert!(narrow_to_i8(-128).is_ok());
    assert!(narrow_to_i8(-129).is_err());
}

/// Tries to cast the sample to a 16-bit signed integer, returning an error on overflow.
fn narrow_to_i16(x: i32) -> Result<i16> {
    use std::i16;
    if x < i16::MIN as i32 || x > i16::MAX as i32 {
        Err(Error::TooWide)
    } else {
        Ok(x as i16)
    }
}

#[test]
fn verify_narrow_to_i16() {
    assert!(narrow_to_i16(32767).is_ok());
    assert!(narrow_to_i16(32768).is_err());
    assert!(narrow_to_i16(-32768).is_ok());
    assert!(narrow_to_i16(-32769).is_err());
}

/// Tries to cast the sample to a 24-bit signed integer, returning an error on overflow.
fn narrow_to_i24(x: i32) -> Result<i32> {
    if x < -(1 << 23) || x > (1 << 23) - 1 {
        Err(Error::TooWide)
    } else {
        Ok(x)
    }
}

#[test]
fn verify_narrow_to_i24() {
    assert!(narrow_to_i24(8_388_607).is_ok());
    assert!(narrow_to_i24(8_388_608).is_err());
    assert!(narrow_to_i24(-8_388_608).is_ok());
    assert!(narrow_to_i24(-8_388_609).is_err());
}

impl Sample for i8 {
    fn write<W: io::Write>(self, writer: &mut W, bits: u16) -> Result<()> {
        match bits {
            8 => Ok(try!(writer.write_u8(u8_from_signed(self)))),
            16 => Ok(try!(writer.write_le_i16(self as i16))),
            24 => Ok(try!(writer.write_le_i24(self as i32))),
            32 => Ok(try!(writer.write_le_i32(self as i32))),
            _ => Err(Error::Unsupported)
        }
    }

    fn read<R: io::Read>(reader: &mut R, fmt: SampleFormat, bytes: u16, bits: u16) -> Result<i8> {
        if fmt != SampleFormat::Int {
            return Err(Error::InvalidSampleFormat);
        }
        match (bytes, bits) {
            (1, 8) => Ok(try!(reader.read_u8().map(signed_from_u8))),
            (n, _) if n > 1 => Err(Error::TooWide),
            // TODO: add a genric decoder for any bit depth.
            _ => Err(Error::Unsupported)
        }
    }
}

impl Sample for i16 {
    fn write<W: io::Write>(self, writer: &mut W, bits: u16) -> Result<()> {
        match bits {
            8 => Ok(try!(writer.write_u8(u8_from_signed(try!(narrow_to_i8(self as i32)))))),
            16 => Ok(try!(writer.write_le_i16(self))),
            24 => Ok(try!(writer.write_le_i24(self as i32))),
            32 => Ok(try!(writer.write_le_i32(self as i32))),
            _ => Err(Error::Unsupported)
        }
    }

    fn read<R: io::Read>(reader: &mut R, fmt: SampleFormat, bytes: u16, bits: u16) -> Result<i16> {
        if fmt != SampleFormat::Int {
            return Err(Error::InvalidSampleFormat);
        }
        match (bytes, bits) {
            (1, 8) => Ok(try!(reader.read_u8().map(signed_from_u8).map(|x| x as i16))),
            (2, 16) => Ok(try!(reader.read_le_i16())),
            (n, _) if n > 2 => Err(Error::TooWide),
            // TODO: add a generic decoder for any bit depth.
            _ => Err(Error::Unsupported)
        }
    }
}

impl Sample for i32 {
    fn write<W: io::Write>(self, writer: &mut W, bits: u16) -> Result<()> {
        match bits {
            8 => Ok(try!(writer.write_u8(u8_from_signed(try!(narrow_to_i8(self)))))),
            16 => Ok(try!(writer.write_le_i16(try!(narrow_to_i16(self))))),
            24 => Ok(try!(writer.write_le_i24(try!(narrow_to_i24(self))))),
            32 => Ok(try!(writer.write_le_i32(self))),
            _ => Err(Error::Unsupported)
        }
    }

    fn read<R: io::Read>(reader: &mut R, fmt: SampleFormat, bytes: u16, bits: u16) -> Result<i32> {
        if fmt != SampleFormat::Int {
            return Err(Error::InvalidSampleFormat);
        }
        match (bytes, bits) {
            (1, 8) => Ok(try!(reader.read_u8().map(signed_from_u8).map(|x| x as i32))),
            (2, 16) => Ok(try!(reader.read_le_i16().map(|x| x as i32))),
            (3, 24) => Ok(try!(reader.read_le_i24())),
            (4, 32) => Ok(try!(reader.read_le_i32())),
            (n, _) if n > 4 => Err(Error::TooWide),
            // TODO: add a generic decoder for any bit depth.
            _ => Err(Error::Unsupported)
        }
    }
}

impl Sample for f32 {
    fn write<W: io::Write>(self, writer: &mut W, bits: u16) -> Result<()> {
        match bits {
            32 => Ok(try!(writer.write_le_f32(self))),
            _ => Err(Error::Unsupported),
        }
    }

    fn read<R: io::Read>(reader: &mut R, fmt: SampleFormat, bytes: u16, bits: u16) -> Result<Self> {
        if fmt != SampleFormat::Float {
            return Err(Error::InvalidSampleFormat);
        }
        match (bytes, bits) {
            (4, 32) => Ok(try!(reader.read_le_f32())),
            (n, _) if n > 4 => Err(Error::TooWide),
            _ => Err(Error::Unsupported),
        }
    }
}

/// Specifies whether a sample is stored as an "IEEE Float" or an integer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampleFormat {
    /// Wave files with the `WAVE_FORMAT_IEEE_FLOAT` format tag store samples as floating point
    /// values.
    ///
    /// Values are normally in the range [-1.0, 1.0].
    Float,
    /// Wave files with the `WAVE_FORMAT_PCM` format tag store samples as integer values.
    Int,
}

/// Specifies properties of the audio data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
    pub bits_per_sample: u16,

    /// Whether the wav's samples are float or integer values.
    pub sample_format: SampleFormat,
}

/// The error type for operations on `WavReader` and `WavWriter`.
#[derive(Debug)]
pub enum Error {
    /// An IO error occured in the underlying reader or writer.
    IoError(io::Error),
    /// Ill-formed WAVE data was encountered.
    FormatError(&'static str),
    /// The sample has more bits than the destination type.
    ///
    /// When iterating using the `samples` iterator, this means that the
    /// destination type (produced by the iterator) is not wide enough to hold
    /// the sample. When writing, this means that the sample cannot be written,
    /// because it requires more bits than the bits per sample specified.
    TooWide,
    /// The number of samples written is not a multiple of the number of channels.
    UnfinishedSample,
    /// The format is not supported.
    Unsupported,
    /// The sample format is different than the destination format.
    ///
    /// When iterating using the `samples` iterator, this means the destination
    /// type (produced by the iterator) has a different sample format than the
    /// samples in the wav file.
    ///
    /// For example, this will occur if the user attempts to produce `i32`
    /// samples (which have a `SampleFormat::Int`) from a wav file that
    /// contains floating point data (`SampleFormat::Float`).
    InvalidSampleFormat,
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter)
           -> result::Result<(), fmt::Error> {
        match *self {
            Error::IoError(ref err) => err.fmt(formatter),
            Error::FormatError(reason) => {
                try!(formatter.write_str("Ill-formed WAVE file: "));
                formatter.write_str(reason)
            },
            Error::TooWide => {
                formatter.write_str("The sample has more bits than the destination type.")
            },
            Error::UnfinishedSample => {
                formatter.write_str("The number of samples written is not a multiple of the number of channels.")
            },
            Error::Unsupported => {
                formatter.write_str("The wave format of the file is not supported.")
            },
            Error::InvalidSampleFormat => {
                formatter.write_str("The sample format differs from the destination format.")
            },
        }
    }
}

impl error::Error for Error {
    fn description(&self) -> &str {
        match *self {
            Error::IoError(ref err) => err.description(),
            Error::FormatError(reason) => reason,
            Error::TooWide => "the sample has more bits than the destination type",
            Error::UnfinishedSample => "the number of samples written is not a multiple of the number of channels",
            Error::Unsupported => "the wave format of the file is not supported",
            Error::InvalidSampleFormat => "the sample format differs from the destination format",
        }
    }

    fn cause(&self) -> Option<&error::Error> {
        match *self {
            Error::IoError(ref err) => Some(err),
            Error::FormatError(_) => None,
            Error::TooWide => None,
            Error::UnfinishedSample => None,
            Error::Unsupported => None,
            Error::InvalidSampleFormat => None,
        }
    }
}

impl From<io::Error> for Error {
    fn from(err: io::Error) -> Error {
        Error::IoError(err)
    }
}

/// A type for results generated by Hound where the error type is hard-wired.
pub type Result<T> = result::Result<T, Error>;

// The WAVEFORMATEXTENSIBLE struct can contain several subformats.
// These are identified by a GUID. The various GUIDS can be found in the file
// mmreg.h that is part of the Windows SDK. The following GUIDS are defined:
// - PCM:        00000001-0000-0010-8000-00aa00389b71
// - IEEE_FLOAT: 00000003-0000-0010-8000-00aa00389b71
// When written to a wav file, the byte order of a GUID is native for the first
// three sections, which is assumed to be little endian, and big endian for the
// last 8-byte section (which does contain a hyphen, for reasons unknown to me).

/// Subformat type for PCM audio with integer samples.
const KSDATAFORMAT_SUBTYPE_PCM: [u8; 16] = [0x01, 0x00, 0x00, 0x00,
                                            0x00, 0x00, 0x10, 0x00,
                                            0x80, 0x00, 0x00, 0xaa,
                                            0x00, 0x38, 0x9b, 0x71];

/// Subformat type for IEEE_FLOAT audio with float samples.
const KSDATAFORMAT_SUBTYPE_IEEE_FLOAT: [u8; 16] = [0x03, 0x00, 0x00, 0x00,
                                                   0x00, 0x00, 0x10, 0x00,
                                                   0x80, 0x00, 0x00, 0xaa,
                                                   0x00, 0x38, 0x9b, 0x71];

#[test]
fn write_read_i16_is_lossless() {
    let mut buffer = io::Cursor::new(Vec::new());
    let write_spec = WavSpec {
        channels: 2,
        sample_rate: 44100,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };

    {
        let mut writer = WavWriter::new(&mut buffer, write_spec);
        for s in -1024_i16 .. 1024 {
            writer.write_sample(s).unwrap();
        }
        writer.finalize().unwrap();
    }

    {
        buffer.set_position(0);
        let mut reader = WavReader::new(&mut buffer).unwrap();
        assert_eq!(write_spec, reader.spec());
        for (expected, read) in (-1024_i16 .. 1024).zip(reader.samples()) {
            assert_eq!(expected, read.unwrap());
        }
    }
}

#[test]
fn write_read_i8_is_lossless() {
    let mut buffer = io::Cursor::new(Vec::new());
    let write_spec = WavSpec {
        channels: 16,
        sample_rate: 48000,
        bits_per_sample: 8,
        sample_format: SampleFormat::Int,
    };

    // Write `i8` samples.
    {
        let mut writer = WavWriter::new(&mut buffer, write_spec);
        // Iterate over i16 because we cannot specify the upper bound otherwise.
        for s in -128_i16 .. 127 + 1 {
            writer.write_sample(s as i8).unwrap();
        }
        writer.finalize().unwrap();
    }

    // Then read them into `i16`.
    {
        buffer.set_position(0);
        let mut reader = WavReader::new(&mut buffer).unwrap();
        assert_eq!(write_spec, reader.spec());
        for (expected, read) in (-128_i16 .. 127 + 1).zip(reader.samples()) {
            assert_eq!(expected, read.unwrap());
        }
    }
}

#[test]
fn write_read_i24_is_lossless() {
    let mut buffer = io::Cursor::new(Vec::new());
    let write_spec = WavSpec {
        channels: 16,
        sample_rate: 96000,
        bits_per_sample: 24,
        sample_format: SampleFormat::Int,
    };

    // Write `i32` samples, but with at most 24 bits per sample.
    {
        let mut writer = WavWriter::new(&mut buffer, write_spec);
        for s in -128_i32 .. 127 + 1 {
            writer.write_sample(s * 256 * 256).unwrap();
        }
        writer.finalize().unwrap();
    }

    // Then read them into `i32`. This should extend the sign in the correct
    // manner.
    {
        buffer.set_position(0);
        let mut reader = WavReader::new(&mut buffer).unwrap();
        assert_eq!(write_spec, reader.spec());
        for (expected, read) in (-128_i32 .. 127 + 1).map(|x| x * 256 * 256)
                                                     .zip(reader.samples()) {
            assert_eq!(expected, read.unwrap());
        }
    }
}
