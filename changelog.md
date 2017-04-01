Changelog
=========

3.0.1
-----

This release fixes a few bugs discovered through fuzzing.

- Fixes high memory usage issue that could occur when reading unknown blocks.
- Resolve various division by zero and arithmetic overflow errors.
- Ensures compatibility with Rust 1.4 through 1.16 stable.

3.0.0
-----

This release focuses on improving write performance. Highlights:

- The header is now written when a `WavWriter` is constructed, therefore
  the constructor now returns a `Result`. This is a breaking change.
- `WavWriter` no longer maintains a buffer internally.
  `WavWriter::create()` does still wrap the file it opens in a buffered
  writer.
- Adds `SampleWriter16` for fast writing of 16-bit samples. Dedicated
  writers for other bit depths might be added in future releases.

Upgrading requires dealing with the `Result` in `WavWriter::new()`
and `WavWriter::create()`. In many cases this should be as simple as
wrapping the call in a `try!()`, or appending a `?` on recent versions
of Rust.

2.0.0
-----

Release highlights:

- Ensures compatibility with Rust 1.4 through 1.10.
- Support for Rust 1.0 through 1.3 has been dropped.
- Adds support for reading files with 32-bit IEEE float samples.

Apart from requiring a newer Rust version, this release includes a minor
breaking change: the `WavSpec` struct gained a new `sample_format`
member. To upgrade, add `sample_format: hound::SampleFormat::Int` to
places where a `WavSpec` is constructed.

Many thanks to Mitchell Nordine for his contributions to this release.

1.1.0
-----

Release highlights:

- New `WavReader::into_inner` method for consistency with the standard library.
- New `WavReader::into_samples` method for ergonomics and consistency.
- Ensures compatibility with Rust 1.4.

Many thanks to Pierre Krieger for his contributions to this release.

1.0.0
-----

This is the first stable release of Hound. Only small changes have been made
with respect to v0.4.0. Release highlights:

- `WavWriter::create` now wraps the writer in a `BufWriter`.
- `WavSamples` now implements `ExactSizeIterator`.
- `WavReader::spec` now returns the spec by value.
- Internal cleanups

0.4.0
-----

Release highlights:

- Works with Rust 1.0.0.
- Hound can now read and write files with 8, 16, 24, or 32 bits per sample.
- Better error reporting
- Improved documentation
- An improved test suite

0.3.0
-----

Release highlights:

- Hound can now read WAVEFORMATEXTENSIBLE, so it can read the files it writes.
- Hound can now read files with PCMWAVEFORMAT and WAVEFORMATEX header.
- Hound now uses a custom error type.
- New convenient filename-based constructors for `WavReader` and `WavWriter`.
- More examples
- An improved test suite

0.2.0
-----
This version adds support for decoding wav files in addition to writing them.

0.1.0
-----

Initial release with only write support.
