use std::io::Write;
use std::time::Duration;

use std::thread;

use opencv::core::{find_file, rotate, Mat, Point, Scalar, Size, ROTATE_180};
use opencv::highgui::{imshow, named_window, wait_key};
use opencv::imgproc::{ellipse, put_text, FONT_HERSHEY_PLAIN};
use opencv::objdetect::CascadeClassifier;
use opencv::prelude::*;
use opencv::types::VectorOfRect;
use opencv::videoio::VideoCaptureTrait;
use serial::prelude::*;
use serial::unix::TTYPort;

struct mode {
    maunal: bool,
}

fn main() {
    let mut mode = mode { maunal: false };

    let mut port = serial::open("/dev/ttyACM0").unwrap();
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

        if frame.empty() {
            break;
        }
        put_text(
            &mut rotated,
            "Press ESC to exit",
            Point::new(10, 30),
            FONT_HERSHEY_PLAIN,
            2.0,
            Scalar::new(0.0, 255.0, 0.0, 0.0),
            2,
            8,
            false,
        )
        .unwrap();

        if mode.maunal == false {
            put_text(
                &mut rotated,
                "AUTO",
                Point::new(10, 55),
                FONT_HERSHEY_PLAIN,
                2.0,
                Scalar::new(0.0, 255.0, 0.0, 0.0),
                2,
                8,
                false,
            )
            .unwrap();
        } else if mode.maunal == true {
            put_text(
                &mut rotated,
                "MANUAL",
                Point::new(10, 55),
                FONT_HERSHEY_PLAIN,
                2.0,
                Scalar::new(0.0, 255.0, 0.0, 0.0),
                2,
                8,
                false,
            )
            .unwrap();
        }

        if mode.maunal == false {
            let location = detect_and_display(&mut rotated, &mut face_cascade);
            execute_commands(&pick_command(location.x, location.y), &mut port);
        }

        imshow(window, &mut rotated).unwrap();
        // arrow key binding to move camera manually
        let input = wait_key(10).unwrap();
        if input == 27 {
            break;
        }
        input_handler(input, &mut port, &mut mode);
    }
}

fn pick_command(x: i32, y: i32) -> String {
    // left boundary = 200
    // right boundary = 1000
    // top boundary = 180
    // bottom boundary = 540

    let mut command = String::new();

    if x == 0 && y == 0 {
        command = "await".to_string();
    } else if x > 500 && x < 700 && y > 300 && y < 400 {
        command = "await".to_string();
    } else {
        if x < 500 {
            command = "left".to_string();
        }
        if x > 700 {
            command = "right".to_string();
        }
        if y < 300 {
            command = "up".to_string();
        }
        if y > 400 {
            command = "down".to_string();
        }
    }
    return command;
}

fn input_handler(input: i32, port: &mut TTYPort, mode: &mut mode) {
    if input == -1 {
        return;
    }
    if input == 83 || input == 2 {
        execute_commands("right", port)
    }
    if input == 81 || input == 3 {
        execute_commands("left", port)
    }
    if input == 82 || input == 0 {
        execute_commands("up", port)
    }
    if input == 84 || input == 1 {
        execute_commands("down", port)
    }
    if input == 9 {
        if mode.maunal == false {
            mode.maunal = true;
        } else if mode.maunal == true {
            mode.maunal = false;
        }
    }
}

fn execute_commands(command: &str, port: &mut TTYPort) {
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
    let largest_face = get_location_of_largest_face(faces);
    return largest_face;
}

fn get_location_of_largest_face(faces: VectorOfRect) -> Point {
    let mut largest_face = Point::new(0, 0);
    let mut largest_face_size = 0;
    for face in faces.iter() {
        if face.width * face.height > largest_face_size {
            largest_face = Point::new(face.x + face.width / 2, face.y + face.height / 2);
            largest_face_size = face.width * face.height;
        }
    }
    return largest_face;
}
