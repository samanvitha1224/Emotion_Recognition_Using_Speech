from django.shortcuts import render
import librosa
import soundfile
import os, glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
# Create your views here.
emotions = {
        '01': 'neutral',
        '02': 'calm',
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fearful',
        '07': 'disgust',
        '08': 'surprised'
    }
observed_emotions = ['happy','sad','neutral','surprised']
def open(request):
    return render(request,'SERui/home.html')

def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma:

            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
    return result
def load_data(test_size=0.25):
    x, y = [], []
    for file in glob.glob("G:\speech-emotion-recognition-ravdess-data\\Actor_*\\*.wav"):
        file_name = os.path.basename(file)
        emotion = emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)
def predict(request):
    if request.method == 'POST':
        if 'wav_file' not in request.FILES.dict().keys():
            context = {'empty': 'Please choose a file that is wav format'}
            return render(request, 'SERui/home.html', context)
        # Split the dataset
        context={}
        file=request.FILES['wav_file']

        feature=extract_feature(file,mfcc=True,chroma=True,mel=True)
        x_train, x_test, y_train, y_test = load_data(test_size=0.25)

        # Get the shape of the training and testing datasets
        context['trainshape']=x_train.shape[0]
        context['testshape']=x_test.shape[0]
        # Get the number of features extracted
        context['fe']=x_train.shape[1]
        # Initialize the Multi Layer Perceptron Classifier
        model = MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,),learning_rate='adaptive', max_iter=500)

        # Train the model
        model.fit(x_train, y_train)

        # Predict for the test set
        y_pred = model.predict(x_test)
        y_pre=model.predict([feature])
        context['result']=y_pre
        #Calculate the accuracy of our model
        accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

        # Print the accuracy
        context['accuracy']=accuracy*100
        # print the classification report
        #print(classification_report(y_test, y_pred))
        context['cr']=classification_report(y_test, y_pred)
        return render(request, "SERui/home.html", context)
