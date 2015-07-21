Hound
=====
A WAV encoding and decoding library in Rust.

[![Build Status][ci-img]][ci]
[![Crates.io version][crate-img]][crate]
[Documentation][docs]

Hound can read and write the WAVE audio format, an ubiquitous format for raw,
uncompressed audio. The main motivation to write it was to test
[Claxon][claxon], a FLAC decoding library written in Rust.

Examples
--------
The following example renders a 440 Hz sine wave, and stores it as a mono wav
file with a sample rate of 44.1 kHz and 16 bits per sample.

```rust
use std::f32::consts::PI;
use std::i16;
use hound;

let spec = hound::WavSpec {
    channels: 1,
    sample_rate: 44100,
    bits_per_sample: 16
};
let mut writer = hound::WavWriter::create("sine.wav", spec).unwrap();
for t in (0 .. 44100).map(|x| x as f32 / 44100.0) {
    let sample = (t * 440.0 * 2.0 * PI).sin();
    let amplitude = i16::MAX as f32;
    writer.write_sample((sample * amplitude) as i16).unwrap();
}
```

The file is finalized implicitly when the writer is dropped, call
`writer.finalize()` to observe errors.

The following example computes the root mean square (RMS) of an audio file with
at most 16 bits per sample.

```rust
use hound;

let mut reader = hound::WavReader::open("testsamples/pop.wav").unwrap();
let sqr_sum = reader.samples::<i16>()
                    .fold(0.0, |sqr_sum, s| {
    let sample = s.unwrap() as f64;
    sqr_sum + sample * sample
});
println!("RMS is {}", (sqr_sum / reader.len() as f64).sqrt());
```

Features
--------
|                 | Read                                                    | Write                  |
|-----------------|---------------------------------------------------------|------------------------|
| Format          | `PCMWAVEFORMAT`, `WAVEFORMATEX`, `WAVEFORMATEXTENSIBLE` | `WAVEFORMATEXTENSIBLE` |
| Encoding        | PCM                                                     | PCM                    |
| Bits per sample | 8, 16, 24, 32                                           | 8, 16, 24, 32          |

License
-------
Hound is licensed under the [GNU General Public License version 3][gplv3]
during the initial development phase.

[ci-img]:    https://travis-ci.org/ruud-v-a/hound.svg?branch=master
[ci]:        https://travis-ci.org/ruud-v-a/hound
[crate-img]: http://img.shields.io/crates/v/hound.svg
[crate]:     https://crates.io/crates/hound
[docs]:      https://ruud-v-a.github.io/hound/doc/v0.4.0/hound/
[claxon]:    https://github.com/ruud-v-a/claxon
[gplv3]:     https://www.gnu.org/licenses/gpl.html
