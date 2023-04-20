use opencv::core::{Size, Scalar, Point, Mat, find_file, rotate, ROTATE_180};
use opencv::highgui::{imshow, named_window, wait_key};
use opencv::imgproc::ellipse;
use opencv::objdetect::{CascadeClassifier};
use opencv::prelude::*;
use opencv::videoio::VideoCaptureTrait;
use opencv::types::VectorOfRect;


fn main() {

    let window = "Capture - Face detection";
    let xml = find_file("haarcascades/haarcascade_frontalface_alt.xml", true, false).unwrap();
    let mut face_cascade = CascadeClassifier::new(&xml).unwrap();
    let mut camera = opencv::videoio::VideoCapture::new(0, opencv::videoio::CAP_ANY).unwrap();
    let mut frame = Mat::default();
    named_window(window, opencv::highgui::WINDOW_NORMAL).unwrap();
    loop {
        camera.read(&mut frame).unwrap();
        let mut rotated = Mat::default();
        rotate(&mut frame, &mut rotated, ROTATE_180).unwrap();
        if frame.empty() {
            break;
        }
        detect_and_display(&mut rotated, &mut face_cascade);
        imshow(window, &mut rotated).unwrap();
        if wait_key(10).unwrap() == 27 {
            break;
        }
    }
}

fn detect_and_display(frame: &mut Mat, face_cascade: &mut CascadeClassifier) {
    let mut frame_gray = Mat::default();
    opencv::imgproc::cvt_color(frame, &mut frame_gray, opencv::imgproc::COLOR_BGR2GRAY, 0).unwrap();
    let mut frame_gray_eq = Mat::default();
    opencv::imgproc::equalize_hist(&frame_gray, &mut frame_gray_eq).unwrap();
    let mut faces = VectorOfRect::new();
    face_cascade.detect_multi_scale(&frame_gray_eq, &mut faces, 1.1, 3, 0, Size::new(30, 30), Size::new(0, 0)).unwrap();
    for face in faces.iter() {
        let center = Point::new(face.x + face.width / 2, face.y + face.height / 2);
        let axes = Size::new(face.width / 2, face.height / 2);
        ellipse(frame, center, axes, 0.0, 0.0, 360.0, Scalar::new(255.0, 0.0, 255.0, 0.0), 4, opencv::imgproc::LINE_8, 0).unwrap();
    }
}
