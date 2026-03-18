import time
import cv2
import pytesseract
from llama_cpp import Llama

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

class PocketProfessorAI:

    def __init__(self, model_path):

        self.llm = Llama(
            model_path=model_path,
            n_ctx=512,
            n_threads=4
        )


        self.system_prompt = """
You are Pocket Professor, an AI tutor.

Rules:
- Give ONLY one short sentence.
- Your response MUST be a guiding question.
- Keep it simple and obvious.
- Do NOT give complex or advanced explanations.
- Do NOT use lists or multiple ideas.

Rule:
Start with the most basic, obvious troubleshooting step.

Examples:

Student: Why is my monitor off?
Hint: Did you turn the monitor on?

Student: Why is my computer not working?
Hint: Is your computer plugged in?

Student: Why is my screen black?
Hint: Is your device powered on?
"""


    def capture_image(self):

        camera = cv2.VideoCapture(0)


        if not camera.isOpened():
            print("Error: Could not access camera.")
            return None

        ret, frame = camera.read()

        camera.release()

        if not ret:
            print("Error: Failed to capture image.")
            return None

        filename = "captured_problem.png"
        cv2.imwrite(filename, frame)
        
        print("Image captured:", filename)

        return filename


    def read_text_from_image(self, image_path):

        image = cv2.imread(image_path)
        text = pytesseract.image_to_string(image)
        return text.strip()


    def get_hint_from_camera(self):

        start_time = time.time()
        image_path = self.capture_image()

        if image_path is None:
            return "I couldn't capture an image.", 0

        question = self.read_text_from_image(image_path)

        if question == "":
            return "I couldn't read any text from the image.", 0

        print("\nDetected Text:\n", question)

    def get_hint(self, question):
        start_time = time.time()

        prompt = f"""
    {self.system_prompt}

    Student Question:
    {question}

    Hint:
    """

        output = self.llm(
            prompt,
            max_tokens=20,
            temperature=0.3,
            stop=["\n\n"]
        )

        hint = output["choices"][0]["text"].strip()
        hint = hint.split("\n")[0]
        # if answer is numbered, remove
        if hint.startswith(("1.", "-", "*")):
            hint = hint[2:].strip()

        response_time = time.time() - start_time

        return hint, response_time