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
//! TODO: Add some examples here.

#![warn(missing_docs)]
#![allow(dead_code)] // TODO: Remove for v0.1
#![feature(convert, io)]

use std::fs;
use std::io;
use std::io::{Seek, Write};
use std::marker;
use std::path;

trait WriteExt: io::Write {
    fn write_le_u16(&mut self, x: u16) -> io::Result<()>;
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

trait Sample {
    fn write<W: io::Write>(self, writer: &mut W) -> io::Result<()>;
}

impl Sample for u16 {
    fn write<W: io::Write>(self, writer: &mut W) -> io::Result<()> {
        writer.write_le_u16(self)
    }
}

pub struct WavSpec {
    n_channels: u16,
    sample_rate: u16,
    bits_per_sample: u16
}

struct WavWriter<W, S> {
    writer: W,
    wrote_header: bool,
    phantom_sample: marker::PhantomData<fn(S)>
}

impl<W, S> WavWriter<W, S>
where W: io::Write + io::Seek,
      S: Sample {

    /// Creates a writer that writes the WAVE format to the underlying writer.
    pub fn new(writer: W, spec: WavSpec) -> WavWriter<W, S> {
        unimplemented!();
    }

    /// Creates a writer that writes the WAVE format to a file.
    ///
    /// The file will be overwritten if it exists.
    pub fn create<P: AsRef<path::Path>>(filename: P, spec: WavSpec)
           -> io::Result<WavWriter<fs::File, S>> {
        let file = try!(fs::File::create(filename));
        Ok(WavWriter::new(file, spec))
    }

    /// Writes the RIFF WAVE header
    fn write_header(&mut self) -> io::Result<()> {
        let mut header = [0u8; 44];

        // Write the header in-memory first.
        {
            let mut buffer: io::Cursor<&mut [u8]> = io::Cursor::new(&mut header);
            try!(buffer.write_all("RIFF".as_bytes()));

            // Skip 4 bytes that will be filled with the file size afterwards.
            buffer.seek(io::SeekFrom::Current(4));

            try!(buffer.write_all("WAVE".as_bytes()));
            try!(buffer.write_all("fmt\0".as_bytes()));
            try!(buffer.write_le_u32(16)); // Size of the WAVE header
            try!(buffer.write_le_u16(1));  // PCM encoded audio
            try!(buffer.write_le_u16(2)); // TODO: num channels
            try!(buffer.write_le_u32(44100)); // TODO: sample rate
            try!(buffer.write_le_u32(0)); // TODO: sr * bps * n_chan / 8
            try!(buffer.write_le_u16(16)); // TODO: block align
            try!(buffer.write_le_u32(16)); // TODO: bits per sample
        }

        // Then write the entire header at once.
        try!(self.writer.write_all(&header));

        Ok(())
    }

    pub fn write_sample(sample: S) -> io::Result<()> {
        // TODO
        Ok(())
    }
}
