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

use std::fs;
use std::io;
use std::mem;
use std::io::Write;
use std::path;
use super::{Error, Result, Sample, SampleFormat, WavSpec};
use ::read;
use ::read::FmtKind;

/// Extends the functionality of `io::Write` with additional methods.
///
/// The methods may be used on any type that implements `io::Write`.
pub trait WriteExt: io::Write {
    /// Writes an unsigned 8-bit integer.
    fn write_u8(&mut self, x: u8) -> io::Result<()>;

    /// Writes a signed 16-bit integer in little endian format.
    fn write_le_i16(&mut self, x: i16) -> io::Result<()>;

    /// Writes an unsigned 16-bit integer in little endian format.
    fn write_le_u16(&mut self, x: u16) -> io::Result<()>;

    /// Writes a signed 24-bit integer in little endian format.
    ///
    /// The most significant byte of the `i32` is ignored.
    fn write_le_i24(&mut self, x: i32) -> io::Result<()>;

    /// Writes an unsigned 24-bit integer in little endian format.
    ///
    /// The most significant byte of the `u32` is ignored.
    fn write_le_u24(&mut self, x: u32) -> io::Result<()>;

    /// Writes a signed 32-bit integer in little endian format.
    fn write_le_i32(&mut self, x: i32) -> io::Result<()>;

    /// Writes an unsigned 32-bit integer in little endian format.
    fn write_le_u32(&mut self, x: u32) -> io::Result<()>;

    /// Writes an IEEE float in little endian format.
    fn write_le_f32(&mut self, x: f32) -> io::Result<()>;
}

impl<W> WriteExt for W
    where W: io::Write
{
    #[inline(always)]
    fn write_u8(&mut self, x: u8) -> io::Result<()> {
        let buf = [x];
        self.write_all(&buf)
    }

    #[inline(always)]
    fn write_le_i16(&mut self, x: i16) -> io::Result<()> {
        self.write_le_u16(x as u16)
    }

    #[inline(always)]
    fn write_le_u16(&mut self, x: u16) -> io::Result<()> {
        let mut buf = [0u8; 2];
        buf[0] = (x & 0xff) as u8;
        buf[1] = (x >> 8) as u8;
        self.write_all(&buf)
    }

    #[inline(always)]
    fn write_le_i24(&mut self, x: i32) -> io::Result<()> {
        self.write_le_u24(x as u32)
    }

    #[inline(always)]
    fn write_le_u24(&mut self, x: u32) -> io::Result<()> {
        let mut buf = [0u8; 3];
        buf[0] = ((x >> 00) & 0xff) as u8;
        buf[1] = ((x >> 08) & 0xff) as u8;
        buf[2] = ((x >> 16) & 0xff) as u8;
        self.write_all(&buf)
    }

    #[inline(always)]
    fn write_le_i32(&mut self, x: i32) -> io::Result<()> {
        self.write_le_u32(x as u32)
    }

    #[inline(always)]
    fn write_le_u32(&mut self, x: u32) -> io::Result<()> {
        let mut buf = [0u8; 4];
        buf[0] = ((x >> 00) & 0xff) as u8;
        buf[1] = ((x >> 08) & 0xff) as u8;
        buf[2] = ((x >> 16) & 0xff) as u8;
        buf[3] = ((x >> 24) & 0xff) as u8;
        self.write_all(&buf)
    }

    #[inline(always)]
    fn write_le_f32(&mut self, x: f32) -> io::Result<()> {
        let u = unsafe { mem::transmute(x) };
        self.write_le_u32(u)
    }
}

/// Generates a bitmask with `channels` ones in the least significant bits.
fn channel_mask(channels: u16) -> u32 {
    (0..channels).map(|c| 1 << c).fold(0, |a, c| a | c)
}

#[test]
fn verify_channel_mask() {
    assert_eq!(channel_mask(0), 0);
    assert_eq!(channel_mask(1), 1);
    assert_eq!(channel_mask(2), 3);
    assert_eq!(channel_mask(3), 7);
    assert_eq!(channel_mask(4), 15);
}

/// A writer that accepts samples and writes the WAVE format.
///
/// The writer needs a `WavSpec` that describes the audio properties. Then
/// samples can be written with `write_sample`. Channel data is interleaved.
/// The number of samples written must be a multiple of the number of channels.
/// After all samples have been written, the file must be finalized. This can
/// be done by calling `finalize`. If `finalize` is not called, the file will
/// be finalized upon drop. However, finalization may fail, and without calling
/// `finalize`, such a failure cannot be observed.
pub struct WavWriter<W>
    where W: io::Write + io::Seek
{
    /// Specifies properties of the audio data.
    spec: WavSpec,

    /// The (container) bytes per sample. This is the bit rate / 8 rounded up.
    bytes_per_sample: u16,

    /// The writer that will be written to.
    writer: W,

    /// The number of bytes written to the data section.
    ///
    /// This is an `u32` because WAVE cannot accomodate more data.
    data_bytes_written: u32,

    /// Whether the header has been finalized.
    finalized: bool,

    /// The buffer for the sample writer, which is recycled throughout calls to
    /// avoid allocating frequently.
    sample_writer_buffer: Vec<u8>,

    /// If true, write WAVEFORMATEXTENSIBLE instead of WAVEFORMATEX.
    extensible: bool,
}

impl<W> WavWriter<W>
    where W: io::Write + io::Seek
{
    /// Creates a writer that writes the WAVE format to the underlying writer.
    ///
    /// The underlying writer is assumed to be at offset 0. `WavWriter` employs
    /// *no* buffering internally. It is recommended to wrap the writer in a
    /// `BufWriter` to avoid too many `write` calls. The `create()` constructor
    /// does this automatically.
    ///
    /// This writes parts of the header immediately, hence a `Result` is
    /// returned.
    pub fn new(writer: W, spec: WavSpec) -> Result<WavWriter<W>> {
        let mut writer = WavWriter {
            spec: spec,
            bytes_per_sample: (spec.bits_per_sample as f32 / 8.0).ceil() as u16,
            writer: writer,
            data_bytes_written: 0,
            sample_writer_buffer: Vec::new(),
            finalized: false,
            // Write the older WAVEFORMAT structure if possible, because it is more
            // widely supported. For more than two channels or more than 16 bits per
            // sample, the newer WAVEFORMATEXTENSIBLE is required. See also
            // https://msdn.microsoft.com/en-us/library/ms713497.aspx. We write up
            // to the point where data should be written.
            extensible: spec.channels > 2 || spec.bits_per_sample > 16,
        };

        try!(writer.write_headers());

        Ok(writer)
    }

    /// Writes the RIFF WAVE header, fmt chunk, and data chunk header.
    fn write_headers(&mut self) -> io::Result<()> {
        // Write to an in-memory buffer before writing to the underlying writer.
        let mut header = [0u8; 68];

        {
            let mut buffer = io::Cursor::new(&mut header[..]);

            // Write the headers for the RIFF WAVE format.
            try!(buffer.write_all("RIFF".as_bytes()));

            // Skip 4 bytes that will be filled with the file size afterwards.
            try!(buffer.write_le_u32(0));

            try!(buffer.write_all("WAVE".as_bytes()));
            try!(buffer.write_all("fmt ".as_bytes()));

            match self.extensible {
                true => try!(self.write_waveformatextensible(&mut buffer)),
                false => try!(self.write_waveformatex(&mut buffer)),
            }

            // Finally the header of the "data" chunk. The number of bytes
            // that this will take is not known at this point. The 0 will
            // be overwritten later.
            try!(buffer.write_all("data".as_bytes()));
            try!(buffer.write_le_u32(0));
        }

        let header_len = if self.extensible { 68 } else { 44 };

        self.writer.write_all(&header[..header_len])
    }

    /// Writes the spec as a WAVEFORMAT structure.
    ///
    /// The WAVEFORMAT struct is a subset of both WAVEFORMATEX
    /// and WAVEFORMATEXTENSIBLE.
    fn write_waveformat(&self, buffer: &mut io::Cursor<&mut [u8]>) -> io::Result<()> {
        let spec = &self.spec;
        // The field nChannels.
        try!(buffer.write_le_u16(spec.channels));

        // The field nSamplesPerSec.
        try!(buffer.write_le_u32(spec.sample_rate));
        let bytes_per_sec = spec.sample_rate
                          * self.bytes_per_sample as u32
                          * spec.channels as u32;

        // The field nAvgBytesPerSec;
        try!(buffer.write_le_u32(bytes_per_sec));

        // The field nBlockAlign. Block align * sample rate = bytes per sec.
        try!(buffer.write_le_u16((bytes_per_sec / spec.sample_rate) as u16));

        Ok(())
    }

    /// Writes the content of the fmt chunk as WAVEFORMATEX struct.
    fn write_waveformatex(&mut self, buffer: &mut io::Cursor<&mut [u8]>) -> io::Result<()> {
        // Write the size of the WAVE header chunk.
        try!(buffer.write_le_u32(16));

        // The following is based on the WAVEFORMATEX struct as documented at
        // https://msdn.microsoft.com/en-us/library/ms713497.aspx. See also
        // http://soundfile.sapp.org/doc/WaveFormat/.

        // The field wFormatTag
        match self.spec.sample_format {
            // WAVE_FORMAT_PCM
            SampleFormat::Int => {
                try!(buffer.write_le_u16(1));
            },
            // WAVE_FORMAT_IEEE_FLOAT
            SampleFormat::Float => {
                if self.spec.bits_per_sample == 32 {
                    try!(buffer.write_le_u16(3));
                } else {
                    panic!("Invalid number of bits per sample. \
                           When writing SampleFormat::Float, \
                           bits_per_sample must be 32.");
                }
            },
        };

        try!(self.write_waveformat(buffer));

        // The field wBitsPerSample, the real number of bits per sample.
        try!(buffer.write_le_u16(self.spec.bits_per_sample));

        Ok(())
    }

    /// Writes the contents of the fmt chunk as WAVEFORMATEXTENSIBLE struct.
    fn write_waveformatextensible(&mut self, buffer: &mut io::Cursor<&mut [u8]>) -> io::Result<()> {
        // Write the size of the WAVE header chunk.
        try!(buffer.write_le_u32(40));

        // The following is based on the WAVEFORMATEXTENSIBLE struct, documented
        // at https://msdn.microsoft.com/en-us/library/ms713496.aspx and
        // https://msdn.microsoft.com/en-us/library/ms713462.aspx.

        // The field wFormatTag, value 1 means WAVE_FORMAT_PCM, but we use
        // the slightly more sophisticated WAVE_FORMAT_EXTENSIBLE.
        try!(buffer.write_le_u16(0xfffe));

        try!(self.write_waveformat(buffer));

        // The field wBitsPerSample. This is actually the size of the
        // container, so this is a multiple of 8.
        try!(buffer.write_le_u16(self.bytes_per_sample as u16 * 8));
        // The field cbSize, the number of remaining bytes in the struct.
        try!(buffer.write_le_u16(22));
        // The field wValidBitsPerSample, the real number of bits per sample.
        try!(buffer.write_le_u16(self.spec.bits_per_sample));
        // The field dwChannelMask.
        // TODO: add the option to specify the channel mask. For now, use
        // the default assignment.
        try!(buffer.write_le_u32(channel_mask(self.spec.channels)));

        // The field SubFormat.
        let subformat_guid = match self.spec.sample_format {
            // PCM audio with integer samples.
            SampleFormat::Int => super::KSDATAFORMAT_SUBTYPE_PCM,
            // PCM audio with 32-bit IEEE float samples.
            SampleFormat::Float => {
                if self.spec.bits_per_sample == 32 {
                    super::KSDATAFORMAT_SUBTYPE_IEEE_FLOAT
                } else {
                    panic!("Invalid number of bits per sample. \
                           When writing SampleFormat::Float, \
                           bits_per_sample must be 32.");
                }
            }
        };
        try!(buffer.write_all(&subformat_guid));

        Ok(())
    }

    /// Writes a single sample for one channel.
    ///
    /// WAVE interleaves channel data, so the channel that this writes the
    /// sample to depends on previous writes. This will return an error if the
    /// sample does not fit in the number of bits specified in the `WavSpec`.
    #[inline]
    pub fn write_sample<S: Sample>(&mut self, sample: S) -> Result<()> {
        try!(sample.write(&mut self.writer, self.spec.bits_per_sample));
        self.data_bytes_written += self.bytes_per_sample as u32;
        Ok(())
    }

    /// Create an efficient writer that writes 16-bit integer samples only.
    ///
    /// When it is known what the kind of samples will be, many dynamic checks
    /// can be omitted. Furthermore, this writer employs buffering internally,
    /// which allows omitting return value checks except on flush. The internal
    /// buffer will be sized such that exactly `num_samples` samples can be
    /// written to it, and the buffer is recycled across calls to
    /// `get_i16_writer()` if the previous buffer was sufficiently large.
    ///
    /// # Panics
    ///
    /// Panics if the spec does not match a 16 bits per sample integer format.
    ///
    /// Attempting to write more than `num_samples` samples to the writer will
    /// panic too.
    pub fn get_i16_writer<'s>(&'s mut self,
                              num_samples: u32)
                              -> SampleWriter16<'s, W> {
        if self.spec.sample_format != SampleFormat::Int {
            panic!("When calling get_i16_writer, the sample format must be int.");
        }
        if self.spec.bits_per_sample != 16 {
            panic!("When calling get_i16_writer, the number of bits per sample must be 16.");
        }

        let num_bytes = num_samples as usize * 2;

        if self.sample_writer_buffer.len() < num_bytes {
            // We need a bigger buffer. There is no point in growing the old
            // one, as we are going to overwrite the samples anyway, so just
            // allocate a new one.
            let mut new_buffer = Vec::with_capacity(num_bytes);

            // The potentially garbage memory here will not be exposed: the
            // buffer is only exposed when flushing, but `flush()` asserts that
            // all samples have been written.
            unsafe { new_buffer.set_len(num_bytes); }

            self.sample_writer_buffer = new_buffer;
        }

        SampleWriter16 {
            writer: &mut self.writer,
            buffer: &mut self.sample_writer_buffer[..num_bytes],
            data_bytes_written: &mut self.data_bytes_written,
            index: 0,
        }
    }

    fn update_header(&mut self) -> Result<()> {
        // The header minus magic and 32-bit filesize.
        let header_size = if self.extensible { 64 } else { 40 };

        let file_size = self.data_bytes_written + (header_size - 4);
        try!(self.writer.seek(io::SeekFrom::Start(4)));
        try!(self.writer.write_le_u32(file_size));
        try!(self.writer.seek(io::SeekFrom::Start(header_size as u64)));
        try!(self.writer.write_le_u32(self.data_bytes_written));

        // Signal error if the last sample was not finished, but do so after
        // everything has been written, so that no data is lost, even though
        // the file is now ill-formed.
        if (self.data_bytes_written / self.bytes_per_sample as u32)
            % self.spec.channels as u32 != 0 {
            Err(Error::UnfinishedSample)
        } else {
            Ok(())
        }
    }

    /// Updates the WAVE header and flushes the underlying writer.
    ///
    /// Flush writes the WAVE header to the underlying writer to make the
    /// written bytes a valid wav file, and then flushes the writer. It is still
    /// possible to write more samples after flushing.
    ///
    /// Flush can be used for “checkpointing”. Even if after the flush there is
    /// an IO error or the writing process dies, the file can still be read by a
    /// compliant decoder up to the last flush.
    ///
    /// Note that if the number of samples written is not a multiple of the
    /// channel count, the intermediate wav file will not be valid. In that case
    /// `flush()` will still flush the data and write the (invalid) wav file,
    /// but `Error::UnfinishedSample` will be returned afterwards.
    ///
    /// It is not necessary to call `finalize()` directly after `flush()`, if no
    /// samples have been written after flushing.
    pub fn flush(&mut self) -> Result<()> {
        let current_pos = try!(self.writer.seek(io::SeekFrom::Current(0)));
        try!(self.update_header());
        try!(self.writer.flush());
        try!(self.writer.seek(io::SeekFrom::Start(current_pos)));
        Ok(())
    }

    /// Updates the WAVE header (which requires knowing all samples).
    ///
    /// This method must be called after all samples have been written. If it
    /// is not called, the destructor will finalize the file, but any errors
    /// that occur in the process cannot be observed in that manner.
    pub fn finalize(mut self) -> Result<()> {
        self.finalized = true;
        self.update_header()
    }

    /// Returns information about the WAVE file being written.
    ///
    /// This is the same spec that was passed to `WavWriter::new()`. For a
    /// writer constructed with `WavWriter::append()`, this method returns
    /// the spec of the file being appended to.
    pub fn spec(&self) -> WavSpec {
        self.spec
    }
}

impl<W> Drop for WavWriter<W>
    where W: io::Write + io::Seek
{
    fn drop(&mut self) {
        // If the file was not explicitly finalized (to update the headers), do
        // it in the drop. This can fail, but drop should not panic, so a
        // failure is ignored silently here.
        if !self.finalized {
            let _r = self.update_header();
        }
    }
}

impl WavWriter<io::BufWriter<fs::File>> {
    /// Creates a writer that writes the WAVE format to a file.
    ///
    /// This is a convenience constructor that creates the file, wraps it in a
    /// `BufWriter`, and then constructs a `WavWriter` from it. The file will
    /// be overwritten if it exists.
    pub fn create<P: AsRef<path::Path>>(filename: P,
                                        spec: WavSpec)
                                        -> Result<WavWriter<io::BufWriter<fs::File>>> {
        let file = try!(fs::File::create(filename));
        let buf_writer = io::BufWriter::new(file);
        WavWriter::new(buf_writer, spec)
    }
}

impl<W> WavWriter<W>
    where W: io::Read + io::Write + io::Seek
{
    /// Creates a writer that appends samples to an existing file.
    ///
    /// This first reads the existing header to obtain the spec, then seeks to
    /// the end of the writer. The writer then appends new samples to the end of
    /// the file.
    ///
    /// The underlying writer is assumed to be at offset 0.
    pub fn append(mut writer: W) -> Result<WavWriter<W>> {
        let (spec_ex, data_len) = {
            try!(read::read_wave_header(&mut writer));
            try!(read::read_until_data(&mut writer))
        };

        // If the format tag was either a WAVEFORMATEX or WAVEFORMATEXTENSIBLE
        // struct, then Hound can write it, so we can update the header. But if
        // it was an other format tag that we can read but not write, then bail
        // out, as we would not know how to update the header.
        let is_extensible = match spec_ex.fmt_kind {
            FmtKind::WaveFormat => return Err(Error::Unsupported),
            FmtKind::WaveFormatEx => false,
            FmtKind::WaveFormatExtensible => true,
        };

        let spec = spec_ex.spec;
        let num_samples = data_len / spec_ex.bytes_per_sample as u32;

        // The number of samples must be a multiple of the number of channels,
        // otherwise the last inter-channel sample would not have data for all
        // channels.
        if num_samples % spec_ex.spec.channels as u32 != 0 {
            return Err(Error::FormatError("invalid data chunk length"));
        }

        try!(writer.seek(io::SeekFrom::Current(data_len as i64)));

        let writer = WavWriter {
            spec: spec,
            bytes_per_sample: spec_ex.bytes_per_sample,
            writer: writer,
            data_bytes_written: data_len,
            sample_writer_buffer: Vec::new(),
            finalized: false,
            extensible: is_extensible,
        };

        Ok(writer)
    }
}


/// A writer that specifically only writes integer samples of 16 bits per sample.
///
/// The writer buffers written samples internally so they can be written in a
/// single batch later on. This has two advantages when performance is
/// important:
///
///  * There is no need for error handling during writing, only on flush. This
///    eliminates a lot of branches.
///  * The buffer can be written once, which reduces the overhead of the write
///    call. Because writing to an `io::BufWriter` is implemented with a
///    `memcpy` (even for single bytes), there is a large overhead to writing
///    small amounts of data such as a 16-bit sample. By writing large blocks
///    (or by not using `BufWriter`) this overhead can be avoided.
///
/// A `SampleWriter16` can be obtained by calling [`WavWriter::get_i16_writer`](
/// struct.WavWriter.html#method.get_i16_writer).
pub struct SampleWriter16<'parent, W> where W: io::Write + io::Seek + 'parent {
    /// The writer borrowed from the wrapped WavWriter.
    writer: &'parent mut W,

    /// The internal buffer that samples are written to before they are flushed.
    buffer: &'parent mut [u8],

    /// Reference to the `data_bytes_written` field of the writer.
    data_bytes_written: &'parent mut u32,

    /// The index into the buffer where the next bytes will be written.
    index: u32,
}

impl<'parent, W: io::Write + io::Seek> SampleWriter16<'parent, W> {
    /// Writes a single sample for one channel.
    ///
    /// WAVE interleaves channel data, so the channel that this writes the
    /// sample to depends on previous writes.
    ///
    /// Unlike `WavWriter::write_sample()`, no range check is performed. Only
    /// the least significant 16 bits are considered, everything else is
    /// discarded.  Apart from that check, this method is more efficient than
    /// `WavWriter::write_sample()`, because it can avoid dispatching on the
    /// number of bits. That was done already when the `SampleWriter16` was
    /// constructed.
    ///
    /// Note that nothing is actually written until `flush()` is called.
    #[inline(always)]
    pub fn write_sample<S: Sample>(&mut self, sample: S) {
        assert!((self.index as usize) <= self.buffer.len() - 2,
          "Trying to write more samples than reserved for the sample writer.");

        let s = sample.as_i16() as u16;

        // Write the sample in little endian to the buffer.
        self.buffer[self.index as usize] = s as u8;
        self.buffer[self.index as usize + 1] = (s >> 8) as u8;

        self.index += 2;
    }

    #[cfg(target_arch = "x86_64")]
    unsafe fn write_u16_le_unchecked(&mut self, value: u16) {
        // x86_64 is little endian, so we do not need to shuffle bytes around;
        // we can just store the 16-bit integer in the buffer directly.
        use std::mem;
        let ptr: *mut u16 = mem::transmute(self.buffer.get_unchecked_mut(self.index as usize));
        *ptr = value;
    }

    #[cfg(not(target_arch = "x86_64"))]
    unsafe fn write_u16_le_unchecked(&mut self, value: u16) {
        // Write a sample in little-endian to the buffer, independent of the
        // endianness of the architecture we are running on.
        let idx = self.index as usize;
        *self.buffer.get_unchecked_mut(idx) = value as u8;
        *self.buffer.get_unchecked_mut(idx + 1) = (value >> 8) as u8;
    }

    /// Like `write_sample()`, but does not perform a bounds check when writing
    /// to the internal buffer.
    ///
    /// It is the responsibility of the programmer to ensure that no more
    /// samples are written than allocated when the writer was created.
    #[inline(always)]
    pub unsafe fn write_sample_unchecked<S: Sample>(&mut self, sample: S) {
        self.write_u16_le_unchecked(sample.as_i16() as u16);
        self.index += 2;
    }

    /// Flush the internal buffer to the underlying writer.
    ///
    /// # Panics
    ///
    /// Panics if insufficient samples (less than specified when the writer was
    /// constructed) have been written with `write_sample()`.
    pub fn flush(self) -> Result<()> {
        if self.index as usize != self.buffer.len() {
            panic!("Insufficient samples written to the sample writer.");
        }

        try!(self.writer.write_all(&self.buffer));
        *self.data_bytes_written += self.buffer.len() as u32;
        Ok(())
    }
}

#[test]
fn short_write_should_signal_error() {
    use SampleFormat;

    let mut buffer = io::Cursor::new(Vec::new());

    let write_spec = WavSpec {
        channels: 17,
        sample_rate: 48000,
        bits_per_sample: 8,
        sample_format: SampleFormat::Int,
    };

    // Deliberately write one sample less than 17 * 5.
    let mut writer = WavWriter::new(&mut buffer, write_spec).unwrap();
    for s in 0..17 * 5 - 1 {
        writer.write_sample(s as i16).unwrap();
    }
    let error = writer.finalize().err().unwrap();

    match error {
        Error::UnfinishedSample => {}
        _ => panic!("UnfinishedSample error should have been returned."),
    }
}

#[test]
fn wide_write_should_signal_error() {
    let mut buffer = io::Cursor::new(Vec::new());

    let spec8 = WavSpec {
        channels: 1,
        sample_rate: 44100,
        bits_per_sample: 8,
        sample_format: SampleFormat::Int,
    };
    {
        let mut writer = WavWriter::new(&mut buffer, spec8).unwrap();
        assert!(writer.write_sample(127_i8).is_ok());
        assert!(writer.write_sample(127_i16).is_ok());
        assert!(writer.write_sample(127_i32).is_ok());
        assert!(writer.write_sample(128_i16).is_err());
        assert!(writer.write_sample(128_i32).is_err());
    }

    let spec16 = WavSpec { bits_per_sample: 16, ..spec8 };
    {
        let mut writer = WavWriter::new(&mut buffer, spec16).unwrap();
        assert!(writer.write_sample(32767_i16).is_ok());
        assert!(writer.write_sample(32767_i32).is_ok());
        assert!(writer.write_sample(32768_i32).is_err());
    }

    let spec24 = WavSpec { bits_per_sample: 24, ..spec8 };
    {
        let mut writer = WavWriter::new(&mut buffer, spec24).unwrap();
        assert!(writer.write_sample(8_388_607_i32).is_ok());
        assert!(writer.write_sample(8_388_608_i32).is_err());
    }
}
