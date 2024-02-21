// Hound -- A wav encoding and decoding library in Rust
// Copyright 2024 Ruud van Asseldonk

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// A copy of the License has been included in the root of the repository.

//! Building blocks for working with the WAVE format.
//!
//! The WAVE format consists of a header, _chunks_ of various kinds, and structs
//! and samples in various formats inside those chunks. It is difficult to
//! expose a single interface that satisfies all of the following criteria:
//!
//! * Usable without deep knowledge of the WAVE format
//! * Easy to use for the common case
//! * Comprehensive (enables access to all chunks and all sample formats)
//! * Performant, low-overhead
//!
//! Therefore Hound does not try to satisfy all of the above at once. Instead,
//! we offer two APIs:
//!
//! * Low-level building blocks for working with the WAVE format. These are
//!   comprehensive and enable writing performant code, but not easy to use.
//!   Reading and writing files involves some boilerplate, and some
//!   understanding of how the WAVE format is structured is needed.
//! * A high level API that is easy to use for the common case, built from the
//!   low-level building blocks.
//!
//! This module contains those low-level building blocks. For the higher level
//! API, see the crate root.

use crate::wav::Error::FormatError;

/// The error type for all operations in Hound.
// TODO: Move this to the crate root.
#[derive(Debug)]
pub enum Error {
    /// An IO error occurred in the underlying reader or writer.
    IoError(std::io::Error),
    /// Ill-formed WAVE data was encountered.
    FormatError(&'static str),
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Error::IoError(err)
    }
}

type Result<T> = std::result::Result<T, Error>;

// TODO: Move to a different module?
trait Bytes {
    fn read_4_bytes(&self) -> [u8; 4];
    fn read_le_u32(&self) -> u32;

    fn write_le_u32(&mut self, x: u32);
}

impl Bytes for [u8] {
    #[inline(always)]
    fn read_4_bytes(&self) -> [u8; 4] {
        [self[0], self[1], self[2], self[3]]
    }

    #[inline(always)]
    fn read_le_u32(&self) -> u32 {
        u32::from_le_bytes(self.read_4_bytes())
    }

    #[inline(always)]
    fn write_le_u32(&mut self, x: u32) {
        self[0] = ((x & 0xff) >> 0) as u8;
        self[1] = ((x & 0xffff) >> 8) as u8;
        self[2] = ((x & 0xffffff) >> 16) as u8;
        self[3] = ((x & 0xffffffff) >> 24) as u8;
    }
}

/// The outermost header of a wav file: the RIFF header.
pub struct RiffHeader {
    /// The length in bytes of the data that follows the header.
    ///
    /// This does not include the length of the 8-byte RIFF header, but it does
    /// include the length of the 4-byte WAVE tag. Therefore the size of a wav
    /// file is 8 bytes more than this inner length.
    ///
    /// A value of `u32::MAX` is in some cases used to signal that the data that
    /// follows is a stream of unknown length, rather than a file with a
    /// specific duration. This meaning is non-standard, but nonetheless occurs
    /// in the wild.
    pub inner_len: u32,
}

impl RiffHeader {
    /// Parse the RIFF WAVE header.
    ///
    /// Aside from returning the length of the file, this can be used to test
    /// the magic bytes to see if a file might be a wav file at all.
    #[inline(always)]
    pub fn from_bytes(bytes: [u8; 12]) -> Result<RiffHeader> {
        if &bytes.read_4_bytes() != b"RIFF" {
            return Err(FormatError("Expected RIFF tag."));
        }
        let result = RiffHeader {
            inner_len: bytes[4..].read_le_u32(),
        };
        if &bytes[8..].read_4_bytes() != b"WAVE" {
            return Err(FormatError("Expected WAVE tag."));
        }
        Ok(result)
    }

    /// Serialize the header for writing to a file.
    #[inline(always)]
    pub fn to_bytes(self) -> [u8; 12] {
        let mut result: [u8; 12] = *b"RIFF\0\0\0\0WAVE";
        result[4..].write_le_u32(self.inner_len);
        result
    }

    /// Construct a header with the maximum possible file size.
    ///
    /// See also the note on [`RiffHeader::inner_len`].
    pub fn nonstandard_infinite() -> RiffHeader {
        RiffHeader {
            inner_len: u32::MAX
        }
    }
}
