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

use std::io;
use std::marker;
use super::{Error, Result, Sample, WavSpec};

// TODO: Can this be unified among Hound and Claxon? Copy + Paste is bad, but
// I refuse to use an external crate just to read into an array of bytes, or
// to read an integer. Such functionality should really be in the standard
// library. Practically _every_ program that does IO will need more high-level
// functionality than what the standard library currently provides.
/// Extends the functionality of `io::Read` with additional methods.
///
/// The methods may be used on any type that implements `io::Read`.
pub trait ReadExt: io::Read {
    /// Reads as many bytes as `buf` is long.
    ///
    /// This may issue multiple `read` calls internally. An error is returned
    /// if `read` read 0 bytes before the buffer is full.
    fn read_into(&mut self, buf: &mut [u8]) -> io::Result<()>;

    /// Reads `n` bytes and returns them in a vector.
    fn read_bytes(&mut self, n: usize) -> io::Result<Vec<u8>>;

    /// Reads two bytes and interprets them as a little-endian 16-bit signed integer.
    fn read_le_i16(&mut self) -> io::Result<i16>;

    /// Reads two bytes and interprets them as a little-endian 16-bit unsigned integer.
    fn read_le_u16(&mut self) -> io::Result<u16>;

    /// Reads four bytes and interprets them as a little-endian 32-bit unsigned integer.
    fn read_le_u32(&mut self) -> io::Result<u32>;
}

impl<R> ReadExt for R where R: io::Read {
    fn read_into(&mut self, buf: &mut [u8]) -> io::Result<()> {
        let mut n = 0;
        while n < buf.len() {
            let progress = try!(self.read(&mut buf[n ..]));
            if progress > 0 {
                n += progress;
            } else {
                return Err(io::Error::new(io::ErrorKind::Other,
                                          "Failed to read enough bytes."));
            }
        }
        Ok(())
    }

    fn read_bytes(&mut self, n: usize) -> io::Result<Vec<u8>> {
        let mut buf = Vec::with_capacity(n);
        // TODO: is there a safe alternative that is not crazy like draining
        // a repeat(0u8) iterator?
        unsafe { buf.set_len(n); }
        try!(self.read_into(&mut buf[..]));
        Ok(buf)
    }

    fn read_le_i16(&mut self) -> io::Result<i16> {
        self.read_le_u16().map(|x| x as i16)
    }

    fn read_le_u16(&mut self) -> io::Result<u16> {
        let mut buf = [0u8; 2];
        try!(self.read_into(&mut buf));
        Ok((buf[1] as u16) << 8 | (buf[0] as u16))
    }

    fn read_le_u32(&mut self) -> io::Result<u32> {
        let mut buf = [0u8; 4];
        try!(self.read_into(&mut buf));
        Ok((buf[3] as u32) << 24 | (buf[2] as u32) << 16 |
           (buf[1] as u32) << 8  | (buf[0] as u32) << 0)
    }
}

/// A reader that reads the WAVE format from the underlying reader.
///
/// A `WavReader` is a streaming reader. It reads data from the underlying
/// reader on demand, and it reads no more than strictly necessary. No internal
/// buffering is performed on the underlying reader.
pub struct WavReader<R> {
    /// Specification of the file as found in the fmt chunk.
    spec: WavSpec,

    /// The number of samples in the data chunk.
    ///
    /// The data chunk is limited to a 4 GiB length because its header has a
    /// 32-bit length field. A sample takes at least one byte to store, so the
    /// number of samples is always less than 2^32.
    num_samples: u32,

    /// The number of samples read so far.
    samples_read: u32,

    /// The reader from which the WAVE format is read.
    reader: R
}

/// An iterator that yields samples of type `S` read from a `WavReader`.
pub struct WavSamples<'wr, R, S> where R: 'wr {
    reader: &'wr mut WavReader<R>,
    phantom_sample: marker::PhantomData<S>
}

impl<R> WavReader<R> where R: io::Read {

    /// Reads the RIFF WAVE header, returns the supposed file size.
    fn read_wave_header(reader: &mut R) -> Result<u32> {
        // Every WAVE file starts with the four bytes 'RIFF' and a file length.
        // TODO: the old approach of having a slice on the stack and reading
        // into it is more cumbersome, but also avoids a heap allocation. Is
        // the compiler smart enough to avoid the heap allocation anyway? I
        // would not expect it to be.
        if "RIFF".as_bytes() != &try!(reader.read_bytes(4))[..] {
            return Err(Error::FormatError("no RIFF tag found"));
        }

        // TODO: would this be useful anywhere? Probably not, except for
        // validating files, but do we need to be so strict?
        let file_len = try!(reader.read_le_u32());

        // Next four bytes indicate the file type, which should be WAVE.
        if "WAVE".as_bytes() != &try!(reader.read_bytes(4))[..] {
            // TODO: use custom error type
            return Err(Error::FormatError("no WAVE tag found"));
        }

        Ok(file_len)
    }

    /// Reads the fmt chunk of the file, returns the information it provides.
    fn read_fmt_chunk(reader: &mut R) -> Result<WavSpec> {
        // Then a "fmt " chunk should follow.
        // TODO: is the "fmt " block always the first block? Should this be
        // flexible? It should anyway, so this hardly matters. For now, we
        // expect only an "fmt " block first, and then a "data" block.
        if "fmt ".as_bytes() != &try!(reader.read_bytes(4))[..] {
            return Err(Error::FormatError("no fmt chunk found"));
        }
        let fmt_chunk_len = try!(reader.read_le_u32());

        // A minimum chunk length of at least 16 is assumed. Note: actually,
        // the first 14 bytes contain enough information to0 fully specify the
        // file. I have not encountered a file with a 14-byte fmt section
        // though. If you ever encounter such file, please contact me.
        if fmt_chunk_len < 16 {
            return Err(Error::FormatError("invalid fmt chunk size"));
        }

        // Read the WAVEFORMAT struct, as defined at
        // https://msdn.microsoft.com/en-us/library/ms713498.aspx.
        // ```
        // typedef struct {
        //     WORD  wFormatTag;
        //     WORD  nChannels;
        //     DWORD nSamplesPerSec;
        //     DWORD nAvgBytesPerSec;
        //     WORD  nBlockAlign;
        // } WAVEFORMAT;
        // ```
        // The WAVEFORMATEX struct has two more members, as defined at
        // https://msdn.microsoft.com/en-us/library/ms713497.aspx
        // ```
        // typedef struct {
        //     WORD  wFormatTag;
        //     WORD  nChannels;
        //     DWORD nSamplesPerSec;
        //     DWORD nAvgBytesPerSec;
        //     WORD  nBlockAlign;
        //     WORD  wBitsPerSample;
        //     WORD  cbSize;
        // } WAVEFORMATEX;
        // ```
        // It appears that in either case, the minimal length of the fmt
        // section is 16 bytes, meaning that it does include the
        // `wBitsPerSample` field. (The name is misleading though, because it
        // is the number of bits used to store a sample, not all of the bits
        // need to be valid for all versions of the WAVE format.)
        let format_tag = try!(reader.read_le_u16());
        let n_channels = try!(reader.read_le_u16());
        let n_samples_per_sec = try!(reader.read_le_u32());
        let n_bytes_per_sec = try!(reader.read_le_u32());
        let block_align = try!(reader.read_le_u16());
        let bits_per_sample = try!(reader.read_le_u16());

        // Two of the stored fields are redundant, and may be ignored. We do
        // validate them to fail early for ill-formed files.
        if (bits_per_sample != block_align / n_channels * 8)
        || (n_bytes_per_sec != block_align as u32 * n_samples_per_sec) {
            return Err(Error::FormatError("inconsistent fmt chunk"));
        }

        if format_tag != 1 {
            // TODO: detect the actual tag, and switch to reading WAVEFORMATEX
            // or WAVEFORMATEXTENSIBLE if indicated by the tag.
            return Err(Error::FormatError("invalid or unsupported format tag"));
        }

        // We have read 16 bytes so far. If the fmt chunk is longer, then we
        // could be dealing with WAVEFORMATEX or WAVEFORMATEXTENSIBLE. This is
        // not supported at this point.
        if fmt_chunk_len > 16 {
            panic!("wave format type not implemented yet");
        }

        let spec = WavSpec {
            channels: n_channels,
            sample_rate: n_samples_per_sec,
            bits_per_sample: bits_per_sample as u32
        };

        Ok(spec)
    }

    /// Reads the header of the data chunk and returns its length.
    fn read_data_chunk(reader: &mut R) -> Result<u32> {
        if "data".as_bytes() != &try!(reader.read_bytes(4))[..] {
            return Err(Error::FormatError("no data chunk found"));
        }
        let data_chunk_len = try!(reader.read_le_u32());
        Ok(data_chunk_len)
    }

    // TODO: define a custom error type to report ill-formed files.
    /// Attempts to create a reader that reads the WAVE format.
    ///
    /// The header is read immediately. Reading the data will be done on
    /// demand.
    pub fn new(mut reader: R) -> Result<WavReader<R>> {
        try!(WavReader::read_wave_header(&mut reader));

        // TODO: read chunk header first, then match on the type, and read the
        // chunk, skip it if unknown, or wait if it is the data chunk.
        let spec = try!(WavReader::read_fmt_chunk(&mut reader));
        let data_len = try!(WavReader::read_data_chunk(&mut reader));

        let num_samples = data_len / (spec.bits_per_sample / 8);
        let wav_reader = WavReader {
            spec: spec,
            num_samples: num_samples,
            samples_read: 0,
            reader: reader
        };

        Ok(wav_reader)
    }

    // TODO: Should this return by value instead? A reference is more consistent
    // with Claxon, but the type is only 80 bits, barely larger than a pointer.
    // Is it worth the extra indirection? On the other hand, the indirection
    // is probably optimised away.
    /// Returns information about the WAVE file.
    pub fn spec(&self) -> &WavSpec {
        &self.spec
    }

    /// Returns an iterator over all samples.
    ///
    /// The channel data is is interleaved. The iterator is streaming. That is,
    /// if you call this method once, read a few samples, and call this method
    /// again, the second iterator will not start again from the beginning of
    /// the file, it will continue where the first iterator stopped.
    pub fn samples<'wr, S: Sample>(&'wr mut self) -> WavSamples<'wr, R, S> {
        WavSamples {
            reader: self,
            phantom_sample: marker::PhantomData
        }
    }
}

impl<'wr, R, S> Iterator for WavSamples<'wr, R, S>
where R: io::Read,
      S: Sample {
    type Item = io::Result<S>;

    fn next(&mut self) -> Option<io::Result<S>> {
        let reader = &mut self.reader;
        if reader.samples_read < reader.num_samples {
            reader.samples_read += 1;
            Some(Sample::read(&mut reader.reader, reader.spec.bits_per_sample))
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let samples_left = self.reader.num_samples - self.reader.samples_read;
        (samples_left as usize, Some(samples_left as usize))
    }
}

/// Tests reading the most basic wav file, one with only a WAVEFORMAT struct.
#[test]
fn read_wav_waveformat() {
    use std::fs;
    
    let file = fs::File::open("testsamples/waveformat-16bit-44100Hz-mono.wav")
                        .ok().expect("failed to open file");
    let buf_reader = io::BufReader::new(file);
    let mut wav_reader = WavReader::new(buf_reader)
                                   .ok().expect("failed to read header");

    assert_eq!(wav_reader.spec().channels, 1);
    assert_eq!(wav_reader.spec().sample_rate, 44100);
    assert_eq!(wav_reader.spec().bits_per_sample, 16);

    let samples: Vec<i16> = wav_reader.samples()
                                      .map(|r| r.ok().unwrap())
                                      .collect();

    // The test file has been prepared with these exact four samples.
    assert_eq!(&samples[..], &[2, -3, 5, -7]);
}

#[test]
fn read_wav_waveformat_ex() {
    // TODO: add a test sample that uses WAVEFORMATEX and verify that it can be
    // read properly.
}

#[test]
fn read_wav_waveformat_extensible() {
    // TODO: add a test sample that uses WAVEFORMATEXTENSIBLE (as produced by
    // Hound itself actually, so this should not be too hard), and verify that
    // it can be read properly.
}
