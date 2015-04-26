extern crate hound;

use std::env;

fn main() {
    let fname = env::args().nth(1).expect("no file given");
    let mut reader = hound::WavReader::open(&fname).unwrap();
    let samples: Vec<i16> = reader.samples().map(|s| s.unwrap()).collect();
    let (sig, unsig, n) = samples.iter().fold((0_f64, 0_f64, 0_u32), |(sig, unsig, n), &s| {
        let sample = s;
        let sig = sig + sample as f64;
        let unsig = unsig + (sample as u16) as f64;
        (sig, unsig, n + 1)
    });
    let ms = sig / n as f64;
    let mu = unsig / n as f64;
    println!("mean signed:   {}\nmean unsigned: {}", ms, mu);

    let (dsig, dunsig, n) = samples.iter().fold((0_f64, 0_f64, 0_u32), |(dsig, dunsig, n), &s| {
        let sample = s;
        let ds = sample as f64 - ms;
        let du = (sample as u16) as f64 - mu;
        (dsig + ds * ds, dunsig + du * du, n + 1)
    });
    let ds = dsig / n as f64;
    let du = dunsig / n as f64;
    println!("rms signed:    {}\nrms unsigned:  {}", ds.sqrt(), du.sqrt());
}
