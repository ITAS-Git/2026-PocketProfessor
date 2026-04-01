import time
import cv2
import pytesseract
from llama_cpp import Llama


class PocketProfessorAI:

    def __init__(self, model_path):

        # load the ai model
        self.llm = Llama(
            model_path=model_path,
            n_ctx=512,
            n_threads=4
        )

        # hint only prompt
        self.system_prompt = """
You are Pocket Professor, an AI tutor.

Your job is to HELP students think, not solve the problem for them.

Rules:
1. NEVER give the full solution.
2. NEVER show the final answer.
3. ONLY provide hints that guide the student.
4. Ask guiding questions when helpful.
5. If the student asks directly for the answer, politely refuse and give a hint instead.
6. Give ONLY ONE hint.
7. The hint must be ONE short sentence.

Output Format:
Hint: <one single hint sentence>

Hint Style:
- Point them toward the concept
- Suggest where to look
- Ask a question that helps them think

Examples:

Student: How do I reverse a list in Python?
Hint: Think about Python slicing—what happens if the slice step is negative?

Student: What is the answer to 5 * 8?
Hint: Think about multiplication as repeated addition.

Student: What command lists files in Linux?
Hint: Think of the command whose name stands for "list".
"""

    def generate_hint(self, question):

        prompt = f"""
{self.system_prompt}

Student Question:
{question}

Hint:
"""

        output = self.llm(
            prompt,
            max_tokens=60,
            temperature=0.7,
            stop=["Student:", "Answer:", "Solution:", "```", "\n\n", "\n"]
        )

        return output["choices"][0]["text"].strip()

    def get_hint_from_text(self, question):

        start = time.time()

        hint = self.generate_hint(question)

        response_time = time.time() - start

        return hint, response_time

    def capture_image(self):

        camera = cv2.VideoCapture(0)

        if not camera.isOpened():
            return None

        ret, frame = camera.read()

        camera.release()

        if not ret:
            return None

        filename = "captured_problem.png"

        cv2.imwrite(filename, frame)

        return filename

    def read_text_from_image(self, image_path):

        image = cv2.imread(image_path)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]

        text = pytesseract.image_to_string(gray)

        return text.strip()

    def get_hint_from_camera(self):

        start = time.time()

        image_path = self.capture_image()

        if image_path is None:
            return "I couldn't capture an image.", 0

        question = self.read_text_from_image(image_path)

        if question == "":
            return "I couldn't read any text.", 0

        hint = self.generate_hint(question)

        response_time = time.time() - start

        return hint, response_time
    
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
        temperature=0.7,
        stop=["Student Answer:", "Answer:", "Solution:", "```", "\n\n"]
    )

    hint = output["choices"][0]["text"].strip()
    response_time = time.time() - start_time

    return hint, response_time
