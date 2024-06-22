import sys
import time

def main():
    for i in range(10):
        sys.stdout.write(f"\r{i}")
        time.sleep(1)
        sys.stdout.flush()

if __name__ == "__main__":
    main()