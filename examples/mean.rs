extern crate hound;

use std::env;

fn main() {
    let fname = env::args().nth(1).expect("no file given");
    let mut reader = hound::WavReader::open(fname).unwrap();
    let (sig, unsig, n) = reader.samples::<i16>()
                             .fold((0_f64, 0_f64, 0_u32), |(sig, unsig, n), s| {
        let sample = s.unwrap();
        let sig = sig + sample as f64;
        let unsig = unsig + (sample as u16) as f64;
        (sig, unsig, n + 1)
    });
    println!("mean signed: {}\nmean unsigned: {}", sig / n as f64, unsig / n as f64);

    let mut reader = hound::WavReader::open(fname).unwrap();
    let (dsig, dunsig, n) = reader.samples::<i16>()
}
