Code for interfacing 2 cameras: Thread.py,
code for just one camera :onecam.py

HARDWARE COMPONENTS USED: 
                     Raspberry pi 4 (4 GB RAM) model B , logitech web camera model c270 – 2 , connecting wires , switch board , Relay element, Table Fan

In this project, we have successfully implemented a real-time human detection system using a Raspberry Pi, OpenCV, and TensorFlow. The objective of the project was to monitor humans in an assembly line of a factory and control the fans to optimize power consumption.
By utilizing the power of deep learning and computer vision, we leveraged the pre-trained ResNet50 model to accurately detect human presence in the captured frames from a single camera. The frames were processed in real-time, and if a human was detected, the system displayed the corresponding notification on the screen. The detection process was optimized by utilizing multi-threading, allowing for efficient frame processing and minimal latency.
The implemented system effectively addressed the primary objective of monitoring human presence in the factory assembly line. By integrating the human detection system with fan control, the project aimed to optimize energy management. When humans were consistently detected within a specified time frame, the system recommended running the fans to provide adequate cooling. Conversely, when no humans were detected for a sufficient duration, the system advised turning off the fans to conserve energy.
The project demonstrates the feasibility and practicality of using computer vision and machine learning techniques for human detection in real-time applications The integration of fan control based on human detection provides a valuable contribution to energy conservation and sustainability in industrial environments.

Looking ahead, there are several potential avenues for further improvement and expansion of this project. Some of the possible future directions include:

1.	Multi-camera Support: Extend the system to handle multiple cameras for enhanced coverage and accurate human tracking across larger areas. This would require synchronization and fusion of data from multiple camera sources.
2.	Human Activity Recognition: Incorporate additional machine learning techniques to recognize specific human activities or gestures, allowing for more advanced control and automation within the factory assembly line.
3.	 Anomaly Detection: Extend the system to detect anomalies or safety hazards in the assembly line environment, providing alerts or triggering appropriate actions when abnormal situations are detected.

Overall, the implemented project lays the foundation for a smart and energy-efficient factory environment by utilizing real-time human detection and fan control. With further advancements and enhancements, this system has the potential to significantly contribute to industrial automation, energy management, and worker safety in assembly line settings.
