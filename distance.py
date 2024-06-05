# Import thư viện time và thư viện giao tiếp với GPIO
import time
import RPi.GPIO as GPIO
import motor
# Khai báo sử dụng cách đánh dấu PIN theo BCM
# Có 2 kiểu đánh dấu PIN là BCM và BOARD
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
# Khởi tạo 2 biến chứa GPIO ta sử dụng
GPIO_TRIGGER = 26
GPIO_ECHO = 19

# Thiết lập GPIO nào để gửi tiến hiệu và nhận tín hiệu
GPIO.setup(GPIO_TRIGGER, GPIO.OUT)  # Trigger
GPIO.setup(GPIO_ECHO, GPIO.IN)      # Echo

# Khai báo này ám chỉ việc hiện tại không gửi tín hiệu điện
# qua GPIO này, kiểu kiểu ngắt điện ấy
GPIO.output(GPIO_TRIGGER, False)


# Cái này mình cũng không rõ, nhưng họ bảo là để khởi động cảm biến
time.sleep(0.5)
def get_distance():
    GPIO.output(GPIO_TRIGGER, False)
    time.sleep(0.1)
    # Kích hoạt cảm biến bằng cách ta nháy cho nó tí điện rồi ngắt đi luôn.
    GPIO.output(GPIO_TRIGGER, True)
    time.sleep(0.00001)
    GPIO.output(GPIO_TRIGGER, False)
    # Đánh dấu thời điểm bắt đầu
    start = time.time()
    while GPIO.input(GPIO_ECHO) == 0:
        start = time.time()
    # Bắt thời điểm nhận được tín hiệu từ Echo
    while GPIO.input(GPIO_ECHO) == 1:
        stop = time.time()

    # Thời gian từ lúc gửi tín hiệu
    elapsed = stop - start

    # Khoảng cách mà tín hiệu đã đi qua trong thời gian đó là thời gian
    # nhân với tốc độ âm thanh (34000 cm/s)
    distance = elapsed * 34000

    # Đó là khoảng cách đi và về, nên chia giá trị này cho 2
    distance = distance / 2

    print("Distance : {:.2f} cm".format(distance))
    return distance    
def advoid_obstacle(distance):
    if distance<= 10.00:
        print("Start advoiding obstacle")
        motor.stop()
        motor.turn_left()
        time.sleep(0.1)
        motor.go_straight(100)
        time.sleep(0.2)
        motor.turn_right()
        time.sleep(0.1)
while True:
    try: 
        motor.go_straight(80)
        time.sleep(0.5)
        distance = float(get_distance())
        time.sleep(0.05)
        advoid_obstacle(distance)
        motor.stop()
        time.sleep(2)
    except KeyboardInterrupt:
        GPIO.cleanup()