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

use std::fs;
use std::io;
use std::marker;
use std::path;
use super::{Error, Result, Sample, WavSpec};

/// Extends the functionality of `io::Read` with additional methods.
///
/// The methods may be used on any type that implements `io::Read`.
pub trait ReadExt: io::Read {
    /// Reads as many bytes as `buf` is long.
    ///
    /// This may issue multiple `read` calls internally. An error is returned
    /// if `read` read 0 bytes before the buffer is full.
    //  TODO: There is an RFC proposing a method like this for the standard library.
    fn read_into(&mut self, buf: &mut [u8]) -> io::Result<()>;

    /// Reads `n` bytes and returns them in a vector.
    fn read_bytes(&mut self, n: usize) -> io::Result<Vec<u8>>;

    /// Reads a single byte and interprets it as an 8-bit signed integer.
    fn read_i8(&mut self) -> io::Result<i8>;

    /// Reads a single byte and interprets it as an 8-bit unsigned integer.
    fn read_u8(&mut self) -> io::Result<u8>;

    /// Reads two bytes and interprets them as a little-endian 16-bit signed integer.
    fn read_le_i16(&mut self) -> io::Result<i16>;

    /// Reads two bytes and interprets them as a little-endian 16-bit unsigned integer.
    fn read_le_u16(&mut self) -> io::Result<u16>;

    /// Reads three bytes and interprets them as a little-endian 24-bit signed integer.
    ///
    /// The sign bit will be extended into the most significant byte.
    fn read_le_i24(&mut self) -> io::Result<i32>;

    /// Reads three bytes and interprets them as a little-endian 24-bit unsigned integer.
    ///
    /// The most significant byte will be 0.
    fn read_le_u24(&mut self) -> io::Result<u32>;

    /// Reads four bytes and interprets them as a little-endian 32-bit signed integer.
    fn read_le_i32(&mut self) -> io::Result<i32>;

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
        // We allocate a runtime fixed size buffer, and we are going to read
        // into it, so zeroing or filling the buffer is a waste. This method
        // is safe, because the contents of the buffer are only exposed when
        // they have been overwritten completely by the read.
        let mut buf = Vec::with_capacity(n);
        unsafe { buf.set_len(n); }
        try!(self.read_into(&mut buf[..]));
        Ok(buf)
    }

    fn read_i8(&mut self) -> io::Result<i8> {
        self.read_u8().map(|x| x as i8)
    }

    fn read_u8(&mut self) -> io::Result<u8> {
        let mut buf = [0u8; 1];
        try!(self.read_into(&mut buf));
        Ok(buf[0])
    }

    fn read_le_i16(&mut self) -> io::Result<i16> {
        self.read_le_u16().map(|x| x as i16)
    }

    fn read_le_u16(&mut self) -> io::Result<u16> {
        let mut buf = [0u8; 2];
        try!(self.read_into(&mut buf));
        Ok((buf[1] as u16) << 8 | (buf[0] as u16))
    }

    fn read_le_i24(&mut self) -> io::Result<i32> {
        self.read_le_u24().map(|x|
            // Test the sign bit, if it is set, extend the sign bit into the
            // most significant byte.
            if x & (1 << 23) == 0 {
                x as i32
            } else {
                (x | 0xff_00_00_00) as i32
            }
        )
    }

    fn read_le_u24(&mut self) -> io::Result<u32> {
        let mut buf = [0u8; 3];
        try!(self.read_into(&mut buf));
        Ok((buf[2] as u32) << 16 | (buf[1] as u32) << 8 | (buf[0] as u32))
    }

    fn read_le_i32(&mut self) -> io::Result<i32> {
        self.read_le_u32().map(|x| x as i32)
    }

    fn read_le_u32(&mut self) -> io::Result<u32> {
        let mut buf = [0u8; 4];
        try!(self.read_into(&mut buf));
        Ok((buf[3] as u32) << 24 | (buf[2] as u32) << 16 |
           (buf[1] as u32) << 8  | (buf[0] as u32) << 0)
    }
}

/// The different chunks that a WAVE file can contain.
enum ChunkKind {
    Fmt,
    Data,
    Unknown
}

/// Describes the structure of a chunk in the WAVE file.
struct ChunkHeader {
    pub kind: ChunkKind,
    pub len: u32
}

/// Specifies properties of the audio data, as well as the layout of the stream.
#[derive(Clone, Copy)]
struct WavSpecEx {
    /// The normal information about the audio data.
    ///
    /// Bits per sample here is the number of _used_ bits per sample, not the
    /// number of bits used to _store_ a sample.
    spec: WavSpec,

    /// The number of bytes used to store a sample.
    bytes_per_sample: u16
}

/// A reader that reads the WAVE format from the underlying reader.
///
/// A `WavReader` is a streaming reader. It reads data from the underlying
/// reader on demand, and it reads no more than strictly necessary. No internal
/// buffering is performed on the underlying reader, but this can easily be
/// added by wrapping the reader in an `io::BufReader`. The `open` constructor
/// takes care of this for you.
pub struct WavReader<R> {
    /// Specification of the file as found in the fmt chunk.
    spec: WavSpec,

    /// The number of bytes used to store a sample in the stream.
    bytes_per_sample: u16,

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
///
/// The type `S` must have at least as many bits as the bits per sample of the
/// file, otherwise every iteration will return an error.
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
        if b"RIFF" != &try!(reader.read_bytes(4))[..] {
            return Err(Error::FormatError("no RIFF tag found"));
        }

        // TODO: would this be useful anywhere? Probably not, except for
        // validating files, but do we need to be so strict?
        let file_len = try!(reader.read_le_u32());

        // Next four bytes indicate the file type, which should be WAVE.
        if b"WAVE" != &try!(reader.read_bytes(4))[..] {
            return Err(Error::FormatError("no WAVE tag found"));
        }

        Ok(file_len)
    }

    /// Attempts to read an 8-byte chunk header.
    fn read_chunk_header(reader: &mut R) -> Result<ChunkHeader> {
        let mut kind_str = [0; 4];
        try!(reader.read_into(&mut kind_str));
        let len = try!(reader.read_le_u32());

        let kind = match &kind_str[..] {
            b"fmt " => ChunkKind::Fmt,
            b"data" => ChunkKind::Data,
            _ => ChunkKind::Unknown
        };

        Ok(ChunkHeader { kind: kind, len: len })
    }

    /// Reads the fmt chunk of the file, returns the information it provides.
    fn read_fmt_chunk(reader: &mut R, chunk_len: u32) -> Result<WavSpecEx> {
        // A minimum chunk length of at least 16 is assumed. Note: actually,
        // the first 14 bytes contain enough information to fully specify the
        // file. I have not encountered a file with a 14-byte fmt section
        // though. If you ever encounter such file, please contact me.
        if chunk_len < 16 {
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
        // There is also PCMWAVEFORMAT as defined at
        // https://msdn.microsoft.com/en-us/library/dd743663.aspx.
        // ```
        // typedef struct {
        //   WAVEFORMAT wf;
        //   WORD       wBitsPerSample;
        // } PCMWAVEFORMAT;
        // ```
        // In either case, the minimal length of the fmt section is 16 bytes,
        // meaning that it does include the `wBitsPerSample` field. (The name
        // is misleading though, because it is the number of bits used to store
        // a sample, not all of the bits need to be valid for all versions of
        // the WAVE format.)
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

        // The bits per sample for a WAVEFORMAT struct is the number of bits
        // used to store a sample. Therefore, it must be a multiple of 8.
        if bits_per_sample % 8 != 0 {
           return Err(Error::FormatError("bits per sample is not a multiple of 8"));
        }

        let spec = WavSpec {
            channels: n_channels,
            sample_rate: n_samples_per_sec,
            bits_per_sample: bits_per_sample
        };

        // The different format tag definitions can be found in mmreg.h that is
        // part of the Windows SDK. The vast majority are esoteric vendor-
        // specific formats. We handle only a few. The following values could
        // be of interest:
        // 0x0001: WAVE_FORMAT_PCM
        // 0x0002: WAVE_FORMAT_ADPCM
        // 0x0003: WAVE_FORMAT_IEEE_FLOAT
        // 0xfffe: WAVE_FORMAT_EXTENSIBLE
        match format_tag {
            0x0001 => WavReader::read_wave_format_pcm(reader, chunk_len, spec),
            0xfffe => WavReader::read_wave_format_extensible(reader, chunk_len, spec),
            _ => Err(Error::Unsupported)
        }
    }

    fn read_wave_format_pcm(mut reader: R, chunk_len: u32, spec: WavSpec)
                            -> Result<WavSpecEx> {
        // When there is a PCMWAVEFORMAT struct, the chunk is 16 bytes long.
        // The WAVEFORMATEX structs includes two extra bytes, `cbSize`.
        let is_wave_format_ex = chunk_len == 18;

        if !is_wave_format_ex && chunk_len != 16 {
            return Err(Error::FormatError("unexpected fmt chunk size"));
        }

        if is_wave_format_ex {
            // For WAVE_FORMAT_PCM which we are reading, there should be no
            // extra data, so `cbSize` should be 0.
            let cb_size = try!(reader.read_le_u16());
            if cb_size != 0 {
                return Err(Error::FormatError("unexpected WAVEFORMATEX size"));
            }

            // For WAVE_FORMAT_PCM in WAVEFORMATEX, only 8 or 16 bits per
            // sample are valid according to
            // https://msdn.microsoft.com/en-us/library/ms713497.aspx.
            if spec.bits_per_sample != 8 && spec.bits_per_sample != 16 {
                return Err(Error::FormatError("bits per sample is not 8 or 16"));
            }
        }

        let spec_ex = WavSpecEx {
            spec: spec,
            bytes_per_sample: spec.bits_per_sample / 8
        };
        Ok(spec_ex)
    }

    fn read_wave_format_extensible(mut reader: R, chunk_len: u32, spec: WavSpec)
                                   -> Result<WavSpecEx> {
        // 16 bytes were read already, there must be two more for the `cbSize`
        // field, and `cbSize` itself must be at least 22, so the chunk length
        // must be at least 40.
        if chunk_len < 40 {
            return Err(Error::FormatError("unexpected fmt chunk size"));
        }

        // `cbSize` is the last field of the WAVEFORMATEX struct.
        let cb_size = try!(reader.read_le_u16());

        // `cbSize` must be at least 22, but in this case we assume that it is
        // 22, because we would not know how to handle extra data anyway.
        if cb_size != 22 {
            return Err(Error::FormatError("unexpected WAVEFORMATEXTENSIBLE size"));
        }

        // What follows is the rest of the `WAVEFORMATEXTENSIBLE` struct, as
        // defined at https://msdn.microsoft.com/en-us/library/ms713496.aspx.
        // ```
        // typedef struct {
        //   WAVEFORMATEX  Format;
        //   union {
        //     WORD  wValidBitsPerSample;
        //     WORD  wSamplesPerBlock;
        //     WORD  wReserved;
        //   } Samples;
        //   DWORD   dwChannelMask;
        //   GUID    SubFormat;
        // } WAVEFORMATEXTENSIBLE, *PWAVEFORMATEXTENSIBLE;
        // ```
        let valid_bits_per_sample = try!(reader.read_le_u16());
        let _channel_mask = try!(reader.read_le_u32()); // Not used for now.
        let mut subformat = [0u8; 16];
        try!(reader.read_into(&mut subformat));

        // Several GUIDS are defined. At the moment, only KSDATAFORMAT_SUBTYPE_PCM
        // is supported.
        // TODO: Extract all these constants to a common place.
        if subformat != [0x01, 0x00, 0x00, 0x00,
                         0x00, 0x00, 0x10, 0x00,
                         0x80, 0x00, 0x00, 0xaa,
                         0x00, 0x38, 0x9b, 0x71] {
            return Err(Error::Unsupported);
        }

        let spec_ex = WavSpecEx {
            spec: WavSpec {
                bits_per_sample: valid_bits_per_sample,
                .. spec
            },
            bytes_per_sample: spec.bits_per_sample / 8
        };
        Ok(spec_ex)
    }

    /// Reads chunks until a data chunk is encountered.
    ///
    /// Returns the information from the fmt chunk and the length of the data
    /// chunk in bytes. Afterwards, the reader will be positioned at the first
    /// content byte of the data chunk.
    fn read_until_data(mut reader: R) -> Result<(WavSpecEx, u32)> {
        let mut spec_opt = None;

        loop {
            let header = try!(WavReader::read_chunk_header(&mut reader));
            match header.kind {
                ChunkKind::Fmt => {
                    let spec = try!(WavReader::read_fmt_chunk(&mut reader,
                                                              header.len));
                    spec_opt = Some(spec);
                },
                ChunkKind::Data => {
                    // The "fmt" chunk must precede the "data" chunk. Any
                    // chunks that come after the data chunk will be ignored.
                    if let Some(spec) = spec_opt {
                        return Ok((spec, header.len));
                    } else {
                        return Err(Error::FormatError("missing fmt chunk"));
                    }
                },
                ChunkKind::Unknown => {
                    // Ignore the chunk; skip all of its bytes.
                    // TODO: this could be more efficient by not allocating
                    // space on the heap, reading into it and then dropping it
                    // without use. For now, this solution is simplest. If Seek
                    // is supported we could skip, but that is a stronger bound
                    // than what is required ...
                    try!(reader.read_bytes(header.len as usize));
                }
            }
            // If no data chunk is ever encountered, the function will return
            // via one of the try! macros that return an Err on end of file.
        }
    }

    /// Attempts to create a reader that reads the WAVE format.
    ///
    /// The header is read immediately. Reading the data will be done on
    /// demand.
    pub fn new(mut reader: R) -> Result<WavReader<R>> {
        try!(WavReader::read_wave_header(&mut reader));
        let (spec_ex, data_len) = try!(WavReader::read_until_data(&mut reader));

        let num_samples = data_len / spec_ex.bytes_per_sample as u32;

        // The number of samples must be a multiple of the number of channels,
        // otherwise the last inter-channel sample would not have data for all
        // channels.
        if num_samples % spec_ex.spec.channels as u32 != 0 {
            return Err(Error::FormatError("invalid data chunk length"));
        }

        let wav_reader = WavReader {
            spec: spec_ex.spec,
            bytes_per_sample: spec_ex.bytes_per_sample,
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
    ///
    /// The type `S` must have at least `spec().bits_per_sample` bits,
    /// otherwise every iteration will return an error. All bit depths up to
    /// 32 bits per sample can be decoded into an `i32`, but if you know
    /// beforehand that you will be reading a file with 16 bits per sample, you
    /// can save memory by decoding into an `i16`.
    pub fn samples<'wr, S: Sample>(&'wr mut self) -> WavSamples<'wr, R, S> {
        WavSamples {
            reader: self,
            phantom_sample: marker::PhantomData
        }
    }

    /// Returns the duration of the file in samples.
    ///
    /// The duration is independent of the number of channels. It is expressed
    /// in units of samples. The duration in seconds can be obtained by
    /// dividing this number by the sample rate. The duration is independent of
    /// how many samples have been read already.
    pub fn duration(&self) -> u32 {
        self.num_samples / self.spec.channels as u32
    }

    /// Returns the number of values that the sample iterator will yield.
    ///
    /// The length of the file is its duration (in samples) times the number of
    /// channels. The length is independent of how many samples have been read
    /// already. To get the number of samples left, use `size_hint` on the
    /// `samples()` iterator.
    pub fn len(&self) -> u32 {
        self.num_samples
    }
}

impl WavReader<io::BufReader<fs::File>> {
    /// Attempts to create a reader that reads from the specified file.
    ///
    /// This is a convenience constructor that opens a `File`, wraps it in a
    /// `BufReader` and then constructs a `WavReader` from it.
    pub fn open<P: AsRef<path::Path>>(filename: P)
                -> Result<WavReader<io::BufReader<fs::File>>> {
        let file = try!(fs::File::open(filename));
        let buf_reader = io::BufReader::new(file);
        WavReader::new(buf_reader)
    }
}

impl<'wr, R, S> Iterator for WavSamples<'wr, R, S>
where R: io::Read,
      S: Sample {
    type Item = Result<S>;

    fn next(&mut self) -> Option<Result<S>> {
        let reader = &mut self.reader;
        if reader.samples_read < reader.num_samples {
            reader.samples_read += 1;
            let sample = Sample::read(&mut reader.reader,
                                      reader.bytes_per_sample,
                                      reader.spec.bits_per_sample);
            Some(sample.map_err(Error::from))
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let samples_left = self.reader.num_samples - self.reader.samples_read;
        (samples_left as usize, Some(samples_left as usize))
    }
}

impl<'wr, R, S> ExactSizeIterator for WavSamples<'wr, R, S>
where R: io::Read,
      S: Sample { }

#[test]
fn duration_and_len_agree() {
    let files = &["testsamples/pcmwaveformat-16bit-44100Hz-mono.wav",
                  "testsamples/waveformatex-16bit-44100Hz-stereo.wav",
                  "testsamples/waveformatextensible-32bit-48kHz-stereo.wav"];

    for fname in files {
        let reader = WavReader::open(fname).unwrap();
        assert_eq!(reader.spec().channels as u32 * reader.duration(),
                   reader.len());
    }
}

/// Tests reading a wave file with the PCMWAVEFORMAT struct.
#[test]
fn read_wav_pcm_wave_format_pcm() {
    let mut wav_reader = WavReader::open("testsamples/pcmwaveformat-16bit-44100Hz-mono.wav")
                                   .unwrap();

    assert_eq!(wav_reader.spec().channels, 1);
    assert_eq!(wav_reader.spec().sample_rate, 44100);
    assert_eq!(wav_reader.spec().bits_per_sample, 16);

    let samples: Vec<i16> = wav_reader.samples()
                                      .map(|r| r.unwrap())
                                      .collect();

    // The test file has been prepared with these exact four samples.
    assert_eq!(&samples[..], &[2, -3, 5, -7]);
}

#[test]
fn read_wav_skips_unknown_chunks() {
    // The test samples are the same as without the -extra suffix, but ffmpeg
    // has kindly added some useless chunks in between the fmt and data chunk.
    let files = ["testsamples/pcmwaveformat-16bit-44100Hz-mono-extra.wav",
                 "testsamples/waveformatex-16bit-44100Hz-mono-extra.wav"];

    for file in files.iter() {
        let mut wav_reader = WavReader::open(file)
                                       .unwrap();

        assert_eq!(wav_reader.spec().channels, 1);
        assert_eq!(wav_reader.spec().sample_rate, 44100);
        assert_eq!(wav_reader.spec().bits_per_sample, 16);

        let sample = wav_reader.samples::<i16>().next().unwrap().unwrap();
        assert_eq!(sample, 2);
    }
}

#[test]
fn len_and_size_hint_are_correct() {
    let mut wav_reader = WavReader::open("testsamples/pcmwaveformat-16bit-44100Hz-mono.wav")
                                   .unwrap();

    assert_eq!(wav_reader.len(), 4);

    {
        let mut samples = wav_reader.samples::<i16>();

        assert_eq!(samples.size_hint(), (4, Some(4)));
        samples.next();
        assert_eq!(samples.size_hint(), (3, Some(3)));
    }

    // Reading should not affect the initial length.
    assert_eq!(wav_reader.len(), 4);

    // Creating a new iterator resumes where the previous iterator stopped.
    {
        let mut samples = wav_reader.samples::<i16>();

        assert_eq!(samples.size_hint(), (3, Some(3)));
        samples.next();
        assert_eq!(samples.size_hint(), (2, Some(2)));
    }
}

#[test]
fn size_hint_is_exact() {
    let files = &["testsamples/pcmwaveformat-16bit-44100Hz-mono.wav",
                  "testsamples/waveformatex-16bit-44100Hz-stereo.wav",
                  "testsamples/waveformatextensible-32bit-48kHz-stereo.wav"];

    for fname in files {
        let mut reader = WavReader::open(fname).unwrap();
        let len = reader.len();
        let mut iter = reader.samples::<i32>();
        for i in 0 .. len {
            let remaining = (len - i) as usize;
            assert_eq!(iter.size_hint(), (remaining, Some(remaining)));
            assert!(iter.next().is_some());
        }
        assert!(iter.next().is_none());
    }
}

/// Tests reading a wave file with the WAVEFORMATEX struct.
#[test]
fn read_wav_wave_format_ex_pcm() {
    let mut wav_reader = WavReader::open("testsamples/waveformatex-16bit-44100Hz-mono.wav")
                                   .unwrap();

    assert_eq!(wav_reader.spec().channels, 1);
    assert_eq!(wav_reader.spec().sample_rate, 44100);
    assert_eq!(wav_reader.spec().bits_per_sample, 16);

    let samples: Vec<i16> = wav_reader.samples()
                                      .map(|r| r.unwrap())
                                      .collect();

    // The test file has been prepared with these exact four samples.
    assert_eq!(&samples[..], &[2, -3, 5, -7]);
}

#[test]
fn read_wav_stereo() {
    let mut wav_reader = WavReader::open("testsamples/waveformatex-16bit-44100Hz-stereo.wav")
                                   .unwrap();

    assert_eq!(wav_reader.spec().channels, 2);
    assert_eq!(wav_reader.spec().sample_rate, 44100);
    assert_eq!(wav_reader.spec().bits_per_sample, 16);

    let samples: Vec<i16> = wav_reader.samples()
                                      .map(|r| r.unwrap())
                                      .collect();

    // The test file has been prepared with these exact eight samples.
    assert_eq!(&samples[..], &[2, -3, 5, -7, 11, -13, 17, -19]);

}

#[test]
fn read_wav_8bit() {
    let mut wav_reader = WavReader::open("testsamples/pcmwaveformat-8bit-44100Hz-mono.wav")
                                   .unwrap();

    assert_eq!(wav_reader.spec().bits_per_sample, 8);

    let samples: Vec<i16> = wav_reader.samples()
                                      .map(|r| r.unwrap())
                                      .collect();

    // The test file has been prepared with these exact four samples.
    assert_eq!(&samples[..], &[19, -53, 89, -127]);
}

/// This test sample tests both reading the WAVEFORMATEXTENSIBLE header, and 24-bit samples.
#[test]
fn read_wav_wave_format_extensible_pcm_24bit() {
    let mut wav_reader = WavReader::open("testsamples/waveformatextensible-24bit-192kHz-mono.wav")
                                   .unwrap();

    assert_eq!(wav_reader.spec().channels, 1);
    assert_eq!(wav_reader.spec().sample_rate, 192_000);
    assert_eq!(wav_reader.spec().bits_per_sample, 24);

    let samples: Vec<i32> = wav_reader.samples()
                                      .map(|r| r.unwrap())
                                      .collect();

    // The test file has been prepared with these exact four samples.
    assert_eq!(&samples[..], &[-17, 4_194_319, -6_291_437, 8_355_817]);
}

#[test]
fn read_wav_32bit() {
    let mut wav_reader = WavReader::open("testsamples/waveformatextensible-32bit-48kHz-stereo.wav")
                                   .unwrap();

    assert_eq!(wav_reader.spec().bits_per_sample, 32);

    let samples: Vec<i32> = wav_reader.samples()
                                      .map(|r| r.unwrap())
                                      .collect();

    // The test file has been prepared with these exact four samples.
    assert_eq!(&samples[..], &[19, -229_373, 33_587_161, -2_147_483_497]);
}

#[test]
fn wide_read_should_signal_error() {
    let mut reader24 = WavReader::open("testsamples/waveformatextensible-24bit-192kHz-mono.wav")
                                 .unwrap();

    // Even though we know the first value is 17, and it should fit in an `i8`,
    // a general 24-bit sample will not fit in an `i8`, so this should fail.
    // 16-bit is still not wide enough, but 32-bit should do the trick.
    assert!(reader24.samples::<i8>().next().unwrap().is_err());
    assert!(reader24.samples::<i16>().next().unwrap().is_err());
    assert!(reader24.samples::<i32>().next().unwrap().is_ok());

    let mut reader32 = WavReader::open("testsamples/waveformatextensible-32bit-48kHz-stereo.wav")
                                 .unwrap();

    // In general, 32-bit samples will not fit in anything but an `i32`.
    assert!(reader32.samples::<i8>().next().unwrap().is_err());
    assert!(reader32.samples::<i16>().next().unwrap().is_err());
    assert!(reader32.samples::<i32>().next().unwrap().is_ok());
}
