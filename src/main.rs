use std::io::Write;
use std::time::Duration;

use std::thread;

use opencv::core::{find_file, rotate, Mat, Point, Scalar, Size, ROTATE_180};
use opencv::highgui::{imshow, named_window, wait_key};
use opencv::imgproc::ellipse;
use opencv::objdetect::CascadeClassifier;
use opencv::prelude::*;
use opencv::types::VectorOfRect;
use opencv::videoio::VideoCaptureTrait;
use serial::prelude::*;
use serial::unix::TTYPort;

fn main() {
    let mut port = serial::open("/dev/tty.usbmodem11301").unwrap();
    port.reconfigure(&|settings| {
        settings.set_baud_rate(serial::Baud9600)?;
        settings.set_char_size(serial::Bits8);
        settings.set_parity(serial::ParityNone);
        settings.set_stop_bits(serial::Stop1);
        settings.set_flow_control(serial::FlowNone);
        Ok(())
    })
    .unwrap();
    port.set_timeout(Duration::from_millis(10000)).unwrap();

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

        let height = frame.rows();
        let width = frame.cols();

        println!("height: {} width: {}", height, width);

        if frame.empty() {
            break;
        }
        let location = detect_and_display(&mut frame, &mut face_cascade);

        println!("location x:{} y:{}", location.x, location.y);
        // left boundary = 200
        // right boundary = 1000
        // top boundary = 180
        // bottom boundary = 540

        if location.x < 200 {
            commands("left", &mut port);
        }
        if location.x > 1000 {
            commands("right", &mut port);
        }
        if location.y < 180 {
            commands("up", &mut port);
        }
        if location.y > 540 {
            commands("down", &mut port);
        }
        if location.x > 200 && location.x < 1000 && location.y > 180 && location.y < 540 {
            commands("await", &mut port);
        }

        imshow(window, &mut frame).unwrap();
        if wait_key(10).unwrap() == 27 {
            break;
        }
    }
}

fn commands(command: &str, port: &mut TTYPort) {
    if command == "left" {
        let data_to_send = "left\n".as_bytes();
        port.write_all(data_to_send).unwrap();
        //println!("Sent: {:?}", data_to_send);
    }
    if command == "right" {
        let data_to_send = "right\n".as_bytes();
        port.write_all(data_to_send).unwrap();
        //println!("Sent: {:?}", data_to_send);
    }
    if command == "up" {
        let data_to_send = "up\n".as_bytes();
        port.write_all(data_to_send).unwrap();
        //println!("Sent: {:?}", data_to_send);
    }
    if command == "down" {
        let data_to_send = "down\n".as_bytes();
        port.write_all(data_to_send).unwrap();
        //println!("Sent: {:?}", data_to_send);
    }
    if command == "await" {
        let data_to_send = "await\n".as_bytes();
        port.write_all(data_to_send).unwrap();
        //println!("Sent: {:?}", data_to_send);
    }
}

fn detect_and_display(frame: &mut Mat, face_cascade: &mut CascadeClassifier) -> Point {
    let mut frame_gray = Mat::default();
    opencv::imgproc::cvt_color(frame, &mut frame_gray, opencv::imgproc::COLOR_BGR2GRAY, 0).unwrap();
    let mut frame_gray_eq = Mat::default();
    opencv::imgproc::equalize_hist(&frame_gray, &mut frame_gray_eq).unwrap();
    let mut faces = VectorOfRect::new();
    face_cascade
        .detect_multi_scale(
            &frame_gray_eq,
            &mut faces,
            1.1,
            3,
            0,
            Size::new(30, 30),
            Size::new(0, 0),
        )
        .unwrap();
    for face in faces.iter() {
        let center = Point::new(face.x + face.width / 2, face.y + face.height / 2);
        let axes = Size::new(face.width / 2, face.height / 2);
        ellipse(
            frame,
            center,
            axes,
            0.0,
            0.0,
            360.0,
            Scalar::new(255.0, 0.0, 255.0, 0.0),
            4,
            opencv::imgproc::LINE_8,
            0,
        )
        .unwrap();
    }
    let largestFace = getLocationOflargestFace(faces);
    return largestFace;
}

fn getLocationOflargestFace(faces: VectorOfRect) -> Point {
    let mut largestFace = Point::new(0, 0);
    let mut largestFaceSize = 0;
    for face in faces.iter() {
        if face.width * face.height > largestFaceSize {
            largestFace = Point::new(face.x + face.width / 2, face.y + face.height / 2);
            largestFaceSize = face.width * face.height;
        }
    }
    return largestFace;
}