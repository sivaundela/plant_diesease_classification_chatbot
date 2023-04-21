Software Tools and Emerging Technologies for AI and ML

Final Project

Authors
Graffyndor Team Members:

- Ashwini – C0849084
- Clannon Francis Noronha – C0846110
- Uday Kumar Burra - C0846072
- Siva Krishna Reddy – C0808232

Description
A plant disease classification chatbot is an AI-based system designed to assist individuals in identifying and classifying plant diseases. The chatbot interacts with users through a messaging interface, allowing users to input images of plants and descriptions of symptoms. The chatbot then uses computer vision and machine learning algorithms to analyze the images and provide a diagnosis and treatment recommendation based on the identified disease.

The chatbot utilizes a vast database of images and information related to plant diseases to provide accurate diagnoses. It also continuously learns and updates its knowledge base based on user input and feedback. The chatbot can be accessed through various platforms, such as messaging apps, websites, and mobile apps, making it easily accessible to a broad audience.

The benefits of a plant disease classification chatbot include improved accuracy and efficiency in disease diagnosis, reduced costs and time associated with traditional diagnosis methods, and increased awareness and education on plant diseases. Additionally, the chatbot can be customized to provide specific information and resources for different regions, crops, and languages, making it a versatile tool for farmers, researchers, and plant enthusiasts worldwide.

Dependencies
Installation Required
1. Natural Language Tool Kit (nltk)
2. pickle
3. sklearn
4. Tensorflow
5. keras
6. flask
7. secrets

Executing program
1. Run training.py first to create machine learning model with data and saves to data folder.
2. Then run app.py, which uses model.h5 from data folder. With the help of flask a local machine a chatbot will display in chrome.
3. In chatbot we can ask basic questions like Hi, How are you? to project related question for example. my plant has yellowing of leaves. Chatbot will replies the related diesease