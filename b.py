from os import system
from time import sleep

while True:
    system("python app.py")
    print ("Restarting...")
    sleep(0.2) # 200ms to CTR+C twice