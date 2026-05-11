from flask import Flask, render_template, Response
import cv2
from deepface import DeepFace

app = Flask(__name__)

# Webcam
camera = cv2.VideoCapture(0)

# Detector de emoções
detector = FER()

def gerar_frames():

    while True:

        sucesso, frame = camera.read()

        if not sucesso:
            break

        # Detectar emoções
        resultados = detector.detect_emotions(frame)

        for resultado in resultados:

            x, y, w, h = resultado["box"]

            # Emoções
            emocoes = resultado["emotions"]

            # Emoção principal
            emocao = max(emocoes, key=emocoes.get)

            # Tradução
            traducao = {
                "happy": "Feliz",
                "sad": "Triste",
                "angry": "Com Raiva",
                "neutral": "Neutro",
                "fear": "Medo",
                "surprise": "Surpreso",
                "disgust": "Nojo"
            }

            emocao = traducao.get(emocao, emocao)

            # Quadrado no rosto
            cv2.rectangle(
                frame,
                (x, y),
                (x+w, y+h),
                (0, 255, 0),
                2
            )

            # Texto da emoção
            cv2.putText(
                frame,
                emocao,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        # Converter imagem
        ret, buffer = cv2.imencode('.jpg', frame)

        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame +
            b'\r\n'
        )

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():

    return Response(
        gerar_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == '__main__':
    app.run(debug=True)