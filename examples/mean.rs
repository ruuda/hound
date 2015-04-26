extern crate hound;

use std::env;

fn main() {
    let fname = env::args().nth(1).expect("no file given");
    let mut reader = hound::WavReader::open(&fname).unwrap();
    let samples: Vec<i16> = reader.samples().map(|s| s.unwrap()).collect();

    let (ts, tu, n) = samples.iter().fold((0.0, 0.0, 0.0), |(ts, tu, n), &s| {
        let signed = s as f64;
        let unsigned = (s as u16) as f64;
        (ts + signed, tu + unsigned, n + 1.0)
    });
    let ms = ts / n;
    let mu = tu / n;
    println!("mean signed:   {} (should be 0, deviation is {})", ms, ms.abs());
    println!("mean unsigned: {} (should be 2^16 - 1, deviation is {})", mu, (mu - 32767.0).abs());

    let (ts, tu) = samples.iter().fold((0.0, 0.0), |(ts, tu), &s| {
        let ds = s as f64 - ms;
        let du = (s as u16) as f64 - mu;
        (ts + ds * ds, tu + du * du)
    });
    let rmss = (ts / n).sqrt();
    let rmsu = (tu / n).sqrt();
    println!("rms signed:    {}", rmss);
    println!("rms unsigned:  {}", rmsu);
}
