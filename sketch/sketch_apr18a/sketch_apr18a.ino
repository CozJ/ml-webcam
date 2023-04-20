#include <Servo.h>

Servo main_motor;
Servo left_motor;
Servo right_motor;

int main_target = 90;
int left_target = 90;
int right_target = 90;

enum class STATE {
  LEFT,
  RIGHT,
  UP,
  DOWN,
  AWAIT,
}

State = STATE::AWAIT;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  main_motor.attach(8, 500, 2500);
  left_motor.attach(9, 500, 2500);
  right_motor.attach(10, 500, 2500);
  pinMode(LED_BUILTIN, OUTPUT);
  delay(100);
  main_motor.write(90);
  left_motor.write(90);
  right_motor.write(90);
  delay(1000);
}

void loop() {
  // put your main code here, to run repeatedly:


  digitalWrite(LED_BUILTIN, LOW);

  if (Serial.available() > 0) {
    String incoming = Serial.readStringUntil('\n');
    if (incoming == "left")
    {
      digitalWrite(LED_BUILTIN, HIGH);
      if ( main_target < 175 ) {
        main_target = main_target + 5;
      }
    }
    if (incoming == "right")
    {
      digitalWrite(LED_BUILTIN, HIGH);
      if ( main_target > 5 ) {
        main_target = main_target - 5;
      }
    }
    if (incoming == "up")
    {
      digitalWrite(LED_BUILTIN, HIGH);
      if (right_target < 175 && left_target > 5)
//      {./
        left_target = left_target - 5;
        right_target = right_target + 5;
      }
    }
    if (incoming == "down")
    {
      digitalWrite(LED_BUILTIN, HIGH);
      if (left_target < 175 && right_target > 5)
      {
        left_target = left_target + 5;
        right_target = right_target - 5;
      }
    }
    if (incoming =="await")
    {
      digitalWrite(LED_BUILTIN, HIGH);
    }
  }

  main_motor.write(main_target);
  left_motor.write(left_target);
  right_motor.write(right_target);
}
