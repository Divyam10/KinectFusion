import MeasurementModule
import numpy

def PythonInit():
    MeasurementModule.Init(python_callback)

def python_callback():
    print(f"Python frame received!\n")
    processed_frame = MeasurementModule.PopFrame()

    if processed_frame.validity_mask.__contains__(True):
        print("Valid frame detected\n")

# Call the C++ function with the Python callback
MeasurementModule.Init(python_callback)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    PythonInit()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
