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

use std::fs;
use std::io;
use std::io::Write;
use std::marker;
use std::path;

/// Extends the functionality of `io::Write` with additional methods.
///
/// The methods may be used on any type that implements `io::Write`.
trait WriteExt: io::Write {
    /// Writes a signed 16-bit integer in little endian format.
    fn write_le_i16(&mut self, x: i16) -> io::Result<()>;

    /// Writes an unsigned 16-bit integer in little endian format.
    fn write_le_u16(&mut self, x: u16) -> io::Result<()>;

    /// Writes an unsigned 32-bit integer in little endian format.
    fn write_le_u32(&mut self, x: u32) -> io::Result<()>;
}

impl<W> WriteExt for W where W: io::Write {
    fn write_le_i16(&mut self, x: i16) -> io::Result<()> {
        self.write_le_u16(x as u16)
    }

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

// TODO: Can this be unified among Hound and Claxon? Copy + Paste is bad, but
// I refuse to use an external crate just to read into an array of bytes, or
// to read an integer. Such functionality should really be in the standard
// library. Practically _every_ program that does IO will need more high-level
// functionality than what the standard library currently provides.
/// Extends the functionality of `io::Read` with additional methods.
///
/// The methods may be used on any type that implements `io::Read`.
trait ReadExt: io::Read {
    /// Reads as many bytes as `buf` is long.
    ///
    /// This may issue multiple `read` calls internally. An error is returned
    /// if `read` read 0 bytes before the buffer is full.
    fn read_into(&mut self, buf: &mut [u8]) -> io::Result<()>;

    /// Reads `n` bytes and returns them in a vector.
    fn read_bytes(&mut self, n: usize) -> io::Result<Vec<u8>>;

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

/// Generates a bitmask with `channels` ones in the least significant bits.
fn channel_mask(channels: u16) -> u32 {
    (0 .. channels).map(|c| 1 << c).fold(0, |a, c| a | c)
}

#[test]
fn verify_channel_mask()
{
    assert_eq!(channel_mask(0), 0);
    assert_eq!(channel_mask(1), 1);
    assert_eq!(channel_mask(2), 3);
    assert_eq!(channel_mask(3), 7);
    assert_eq!(channel_mask(4), 15);
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

    /// The (container) bytes per sample. This is the bit rate / 8 rounded up.
    bytes_per_sample: u32,

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
            bytes_per_sample: (spec.bits_per_sample as f32 / 8.0).ceil() as u32,
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
        // Useful links:
        // https://msdn.microsoft.com/en-us/library/ms713462.aspx
        // https://msdn.microsoft.com/en-us/library/ms713497.aspx

        let mut header = [0u8; 68];
        let spec = &self.spec;

        // Write the header in-memory first.
        {
            let mut buffer: io::Cursor<&mut [u8]> = io::Cursor::new(&mut header);
            try!(buffer.write_all("RIFF".as_bytes()));

            // Skip 4 bytes that will be filled with the file size afterwards.
            try!(buffer.write_le_u32(0));

            try!(buffer.write_all("WAVE".as_bytes()));
            try!(buffer.write_all("fmt ".as_bytes()));
            try!(buffer.write_le_u32(40)); // Size of the WAVE header chunk.

            // The following is based on the WAVEFORMATEXTENSIBLE struct as
            // documented on MSDN.

            // The field wFormatTag, value 1 means WAVE_FORMAT_PCM, but we use
            // the slightly more sophisticated WAVE_FORMAT_EXTENSIBLE.
            try!(buffer.write_le_u16(0xfffe));
            // The field nChannels.
            try!(buffer.write_le_u16(spec.channels));
            // The field nSamplesPerSec.
            try!(buffer.write_le_u32(spec.sample_rate));
            let bytes_per_sec = spec.sample_rate
                              * self.bytes_per_sample
                              * spec.channels as u32;
            // The field nAvgBytesPerSec;
            try!(buffer.write_le_u32(bytes_per_sec));
            // The field nBlockAlign. Block align * sample rate = bytes per sec.
            try!(buffer.write_le_u16((bytes_per_sec / spec.sample_rate) as u16));
            // The field wBitsPerSample. This is actually the size of the
            // container, so this is a multiple of 8.
            try!(buffer.write_le_u16(self.bytes_per_sample as u16 * 8));
            // The field cbSize, the number of remaining bytes in the struct.
            try!(buffer.write_le_u16(22));
            // The field wValidBitsPerSample, the real number of bits per sample.
            try!(buffer.write_le_u16(self.spec.bits_per_sample as u16));
            // The field dwChannelMask.
            // TODO: add the option to specify the channel mask. For now, use
            // the default assignment.
            try!(buffer.write_le_u32(channel_mask(self.spec.channels)));
            // The field SubFormat. We use KSDATAFORMAT_SUBTYPE_PCM. The
            // following GUIDS are defined:
            // - PCM:        00000001-0000-0010-8000-00aa00389b71
            // - IEEE_FLOAT: 00000003-0000-0010-8000-00aa00389b71
            // The byte order of a GUID is native for the first three sections,
            // which is assumed to be little endian, and big endian for the
            // last 8-byte section (which does contain a hyphen, for reasons
            // unknown to me).
            try!(buffer.write_all(&[0x01, 0x00, 0x00, 0x00,
                                    0x00, 0x00, 0x10, 0x00,
                                    0x80, 0x00, 0x00, 0xaa,
                                    0x00, 0x38, 0x9b, 0x71]));

            // So far the "fmt " section, now comes the "data" section. We will
            // only write the header here, actual data are the samples. The
            // number of bytes that this will take is not known at this point.
            // The 0 will be overwritten later.
            try!(buffer.write_all("data".as_bytes()));
            try!(buffer.write_le_u32(0));
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
            self.wrote_header = true;
        }

        // TODO: do we need bits per sample? Is the padding at the obvious side?
        try!(sample.write(&mut self.writer, self.bytes_per_sample));
        self.data_bytes_written += self.bytes_per_sample;
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

        // The header minus magic and 32-bit filesize is 60 bytes long.
        let file_size = self.data_bytes_written + 60;
        try!(writer.seek(io::SeekFrom::Start(4)));
        try!(writer.write_le_u32(file_size));
        try!(writer.seek(io::SeekFrom::Start(64)));
        try!(writer.write_le_u32(self.data_bytes_written));

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

// TODO: Add benchmark for write speed.

/// A reader that reads the WAVE format from the underlying reader.
pub struct WavReader<R> {
    /// Specification of the file as found in the fmt chunk.
    spec: WavSpec,

    /// The number of samples in the data chunk.
    ///
    /// The data chunk is limited to a 4 GiB length because its header has a
    /// 32-bit length field. A sample takes at least one byte to store, so the
    /// number of samples is always less than 2^32.
    num_samples: u32,

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
    fn read_wave_header(reader: &mut R) -> io::Result<u32> {
        // Every WAVE file starts with the four bytes 'RIFF' and a file length.
        // TODO: the old approach of having a slice on the stack and reading
        // into it is more cumbersome, but also avoids a heap allocation. Is
        // the compiler smart enough to avoid the heap allocation anyway? I
        // would not expect it to be.
        if "RIFF".as_bytes() != &try!(reader.read_bytes(4))[..] {
            // TODO: use custom error type
            return Err(io::Error::new(io::ErrorKind::Other, "No RIFF tag found."));
        }

        // TODO: would this be useful anywhere? Probably not, except for
        // validating files, but do we need to be so strict?
        let file_len = try!(reader.read_le_u32());

        // Next four bytes indicate the file type, which should be WAVE.
        if "WAVE".as_bytes() != &try!(reader.read_bytes(4))[..] {
            // TODO: use custom error type
            return Err(io::Error::new(io::ErrorKind::Other, "No WAVE tag found."));
        }

        Ok(file_len)
    }

    // TODO: define a custom error type to report ill-formed files.
    /// Reads the fmt chunk of the file, returns the information it provides.
    fn read_fmt_chunk(reader: &mut R) -> io::Result<WavSpec> {
        // Then a "fmt " chunk should follow.
        // TODO: is the "fmt " block always the first block? Should this be
        // flexible? It should anyway, so this hardly matters. For now, we
        // expect only an "fmt " block first, and then a "data" block.
        if "fmt ".as_bytes() != &try!(reader.read_bytes(4))[..] {
            // TODO: use custom error type
            return Err(io::Error::new(io::ErrorKind::Other, "No fmt block found."));
        }
        let fmt_chunk_len = try!(reader.read_le_u32());

        // A minimum chunk length of at least 16 is assumed. Note: actually,
        // the first 14 bytes contain enough information to0 fully specify the
        // file. I have not encountered a file with a 14-byte fmt section
        // though. If you ever encounter such file, please contact me.
        if fmt_chunk_len < 16 {
            // TODO: use custom error type
            return Err(io::Error::new(io::ErrorKind::Other, "Invalid fmt chunck size."));
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
            // TODO: use custom error type
            return Err(io::Error::new(io::ErrorKind::Other, "Incosistent fmt chunk."));
        }

        if format_tag != 1 {
            // TODO: detect the actual tag, and switch to reading WAVEFORMATEX
            // or WAVEFORMATEXTENSIBLE if indicated by the tag.
            // TODO: use a custom error type, report proper error.
            return Err(io::Error::new(io::ErrorKind::Other, "Invalid or unsupported format tag."));
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
    fn read_data_chunk(reader: &mut R) -> io::Result<u32> {
        if "data".as_bytes() != &try!(reader.read_bytes(4))[..] {
            // TODO: use custom error type
            return Err(io::Error::new(io::ErrorKind::Other, "No data block found."));
        }
        let data_chunk_len = try!(reader.read_le_u32());
        Ok(data_chunk_len)
    }

    // TODO: define a custom error type to report ill-formed files.
    /// Attempts to create a reader that reads the WAVE format.
    ///
    /// The header is read immediately. Reading the data will be done on
    /// demand.
    pub fn new(mut reader: R) -> io::Result<WavReader<R>> {
        try!(WavReader::read_wave_header(&mut reader));
        let spec = try!(WavReader::read_fmt_chunk(&mut reader));
        let data_len = try!(WavReader::read_data_chunk(&mut reader));

        let wav_reader = WavReader {
            spec: spec,
            num_samples: data_len / (spec.bits_per_sample / 8),
            reader: reader
        };

        Ok(wav_reader)
    }

    // TODO: Should this return by value instead? A reference is more consistent
    // with Claxon, but the type is only 80 bytes, barely larger than a pointer.
    // Is it worth the extra indirection? On the other hand, the indirection
    // is probably optimised away.
    /// Returns information about the WAVE file.
    pub fn spec(&self) -> &WavSpec {
        &self.spec
    }

    pub fn samples<'wr, S: Sample>(&'wr mut self) -> WavSamples<'wr, R, S> {
        WavSamples {
            reader: self,
            phantom_sample: marker::PhantomData
        }
    }
}

/// Tests reading the most basic wav file, one with only a WAVEFORMAT struct.
#[test]
fn read_wav_waveformat() {
    let file = fs::File::open("testsamples/waveformat.wav")
                        .ok().expect("failed to open file");
    let buf_reader = io::BufReader::new(file);
    let wav_reader = WavReader::new(buf_reader)
                               .ok().expect("failed to read header");

    assert_eq!(wav_reader.spec().channels, 2);
    assert_eq!(wav_reader.spec().sample_rate, 44100);
    assert_eq!(wav_reader.spec().bits_per_sample, 16);

    // TODO: implement Iterator for samples.
    // let samples: Vec<i16> = wav_reader.samples().collect();
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
