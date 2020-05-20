extern crate opencv;

use std::*;

use opencv::{
    core,
    imgproc,
    objdetect,
    prelude::*,
    types
};

fn get_faces(frame: &opencv::prelude::Mat) -> opencv::Result<types::VectorOfRect> {
    let mut gray = Mat::default()?;
    imgproc::cvt_color(frame, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;

    let xml= core::find_file("haarcascades/haarcascade_frontalface_default.xml", true, false)?;
    let mut face = objdetect::CascadeClassifier::new(&xml)?;

    let mut faces = types::VectorOfRect::new();
    face.detect_multi_scale(&gray, &mut faces, 1.1, 3, 0, core::Size{width: 40, height: 40 }, core::Size { width: 0, height: 0 })?;

    return Ok(faces);
}

fn run() -> opencv::Result<()> { 
    let mut frame = opencv::imgcodecs::imread("input-2.jpg", opencv::imgcodecs::IMREAD_COLOR)?;

    let faces = get_faces(&mut frame)?;

    println!("faces: {}", faces.len());
    for face in faces {
        println!("face {:?}", face);
        imgproc::rectangle(&mut frame, face, core::Scalar::new(0f64, 0f64, 255f64, 0f64), 3, 8, 0)?;
    }

    let mut params: core::Vector<i32> = core::Vector::new();
    opencv::imgcodecs::imwrite("output-2.jpg", &frame, &mut params)?;

    return Ok(())
}

fn main() {
    run().unwrap()
}